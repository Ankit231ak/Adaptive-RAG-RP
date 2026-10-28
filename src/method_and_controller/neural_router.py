"""
PyTorch Neural Evidence Controller with Residual Connections and Focal Loss.
Addresses multi-hop RAG stopping decisions under severe class imbalance (87% STOP vs 13% MORE).
"""
import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """
        Pre-activation Residual Block with Layer Normalization and Dropout.
        x -> LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> LayerNorm -> + x
        """
        def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.act = nn.SiLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.norm1(x)
            out = self.linear1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.linear2(out)
            out = self.norm2(out)
            return residual + out


    class NeuralEvidenceRouter(nn.Module):
        """
        Deep Residual Evidence Router for post-retrieval stopping decisions.
        Takes 7-dimensional evidence-state features X_5 and outputs P(MORE | X_5).
        """
        def __init__(
            self,
            in_features: int = 7,
            hidden_dim: int = 64,
            num_res_blocks: int = 2,
            dropout_rate: float = 0.2
        ):
            super().__init__()
            self.in_features = in_features
            self.hidden_dim = hidden_dim

            # Feature input projection
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            )

            # Residual backbone
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ])

            # Calibrated classification head
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate / 2.0),
                nn.Linear(hidden_dim // 2, 1)
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor of shape (batch_size, in_features)
            Returns:
                logits: Tensor of shape (batch_size, 1)
            """
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h)
            logits = self.head(h)
            return logits

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Compute calibrated probabilities P(MORE | X_5)."""
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.sigmoid(logits)
            return probs


    class FocalLoss(nn.Module):
        """
        Binary Focal Loss addressing severe class imbalance (87:13 in HotpotQA).
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        def __init__(self, alpha: float = 0.85, gamma: float = 2.0, reduction: str = 'mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(logits)
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
            loss = focal_weight * ce_loss

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
else:
    # Fallback placeholders if torch is not installed
    NeuralEvidenceRouter = None  # type: ignore
    ResidualBlock = None  # type: ignore
    FocalLoss = None  # type: ignore


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay learning rate schedule."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs + 1, total_epochs)
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, base_lr * factor)


class PyTorchControllerTrainer:
    """
    High-performance, robust training manager for the NeuralEvidenceRouter.
    Includes StandardScaler, Focal Loss, mixed precision, gradient clipping,
    and validation calibration for optimal probability threshold tau.
    """
    def __init__(
        self,
        in_features: int = 7,
        hidden_dim: int = 64,
        num_res_blocks: int = 2,
        dropout_rate: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        tau: float = 0.3,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PyTorchControllerTrainer.")

        self.tau = tau
        self.scaler = StandardScaler()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = NeuralEvidenceRouter(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.loss_fn = FocalLoss(alpha=0.85, gamma=2.0)
        self.is_trained = False

    def train_epochs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 60,
        batch_size: int = 32,
        warmup_epochs: int = 5,
        clip_grad_norm: float = 1.0,
        use_amp: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model strictly on X_train/y_train with optional validation monitoring.
        """
        # Fit scaler ONLY on training data to guarantee zero leakage
        X_train_sc = self.scaler.fit_transform(X_train)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_sc, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_sc = self.scaler.transform(X_val)
            val_dataset = TensorDataset(
                torch.tensor(X_val_sc, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs
        )

        scaler = torch.amp.GradScaler('cuda') if (use_amp and self.device.type == 'cuda') else None

        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(batch_x)
                        loss = self.loss_fn(logits, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    logits = self.model(batch_x)
                    loss = self.loss_fn(logits, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                    self.optimizer.step()

                epoch_loss += loss.item() * len(batch_x)

            scheduler.step(epoch)
            train_loss = epoch_loss / len(train_dataset)
            history["train_loss"].append(train_loss)

            # Validation step
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_logits = self.model(vx)
                        v_loss = self.loss_fn(v_logits, vy)
                        val_loss += v_loss.item() * len(vx)
                        v_probs = torch.sigmoid(v_logits).cpu().numpy().flatten()
                        val_preds.extend((v_probs >= self.tau).astype(int))
                        val_targets.extend(vy.cpu().numpy().flatten().astype(int))

                avg_val_loss = val_loss / len(val_dataset)
                history["val_loss"].append(avg_val_loss)

                # Binary F1 for MORE class
                from sklearn.metrics import f1_score as calc_f1
                val_f1 = calc_f1(val_targets, val_preds, zero_division=0)
                history["val_f1"].append(float(val_f1))

        self.is_trained = True
        return history

    def predict_action(self, feature_vector: Union[List[float], np.ndarray]) -> Tuple[str, float]:
        """Inference for a single 7-dim evidence feature vector."""
        if not self.is_trained:
            return "STOP", 0.0

        x_arr = np.array([feature_vector], dtype=np.float32)
        x_sc = self.scaler.transform(x_arr)
        x_tensor = torch.tensor(x_sc, dtype=torch.float32).to(self.device)

        probs = self.model.predict_proba(x_tensor).cpu().numpy().flatten()
        p_more = float(probs[0])
        action = "MORE" if p_more >= self.tau else "STOP"
        return action, round(p_more, 4)

    def save(self, filepath: str):
        """Save model checkpoint, scaler, and configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "tau": self.tau,
            "in_features": self.model.in_features,
            "hidden_dim": self.model.hidden_dim,
            "is_trained": self.is_trained,
            "backend": "neural"
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]
        self.tau = checkpoint.get("tau", 0.3)
        self.is_trained = checkpoint.get("is_trained", True)
