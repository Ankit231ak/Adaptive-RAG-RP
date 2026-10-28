"""
Adaptive Evidence-State STOP/MORE Controller Router.
Supports both PyTorch Deep Residual Controller and Scikit-learn GBDT backends.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Attempt to import PyTorch neural components
try:
    from src.method_and_controller.neural_router import (
        NeuralEvidenceRouter,
        PyTorchControllerTrainer,
        TORCH_AVAILABLE
    )
except ImportError:
    TORCH_AVAILABLE = False
    NeuralEvidenceRouter = None  # type: ignore
    PyTorchControllerTrainer = None  # type: ignore


class ControllerRouter:
    """
    Binary Evidence Controller predicting P(MORE | X_5).
    Decides STOP (early-exit, generate from D5) or MORE (retrieve D6:10, generate from D10).
    
    Backends:
    - 'neural': PyTorch Deep Residual MLP with LayerNorm, Focal Loss, and Dropout.
    - 'gbdt': Calibrated Gradient Boosting Decision Tree (100 estimators, max depth 3).
    """

    def __init__(self, tau: float = 0.3, backend: str = "gbdt"):
        self.tau = tau
        self.backend = backend.lower()
        self.scaler = StandardScaler()
        self.is_trained = False

        if self.backend == "neural" and TORCH_AVAILABLE:
            self.neural_trainer = PyTorchControllerTrainer(
                in_features=7,
                hidden_dim=64,
                num_res_blocks=2,
                dropout_rate=0.2,
                tau=self.tau
            )
            self.model = self.neural_trainer.model
        else:
            if self.backend == "neural":
                logger.warning("PyTorch not available. Falling back to GBDT backend.")
                self.backend = "gbdt"
            self.neural_trainer = None
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )

    def train(
        self,
        feature_vectors: Union[List[List[float]], np.ndarray],
        labels: Union[List[str], List[int], np.ndarray],
        val_features: Optional[Union[List[List[float]], np.ndarray]] = None,
        val_labels: Optional[Union[List[str], List[int], np.ndarray]] = None,
        epochs: int = 50
    ):
        """
        Train controller strictly on training features and oracle labels.
        Labels can be strings ('STOP' / 'MORE') or integers (0 / 1).
        """
        if len(feature_vectors) == 0:
            return

        X = np.array(feature_vectors, dtype=np.float32)
        if isinstance(labels[0], (int, np.integer)):
            y = np.array(labels, dtype=np.int32)
        else:
            y = np.array([1 if str(lbl).upper() == "MORE" else 0 for lbl in labels], dtype=np.int32)

        if len(np.unique(y)) < 2:
            logger.warning("Single class present in training data. Controller remains untrained.")
            self.is_trained = False
            return

        if self.backend == "neural" and self.neural_trainer is not None:
            X_val_arr = None
            y_val_arr = None
            if val_features is not None and val_labels is not None:
                X_val_arr = np.array(val_features, dtype=np.float32)
                if isinstance(val_labels[0], (int, np.integer)):
                    y_val_arr = np.array(val_labels, dtype=np.int32)
                else:
                    y_val_arr = np.array([1 if str(l).upper() == "MORE" else 0 for l in val_labels], dtype=np.int32)

            self.neural_trainer.train_epochs(
                X_train=X,
                y_train=y,
                X_val=X_val_arr,
                y_val=y_val_arr,
                epochs=epochs
            )
            self.scaler = self.neural_trainer.scaler
            self.is_trained = True
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True

    def predict_action(self, feature_vector: Union[List[float], np.ndarray]) -> Tuple[str, float]:
        """
        Predict STOP or MORE based on probability threshold tau.
        Returns:
            action: 'STOP' or 'MORE'
            p_more: Float probability of requiring expanded retrieval
        """
        if not self.is_trained:
            return "STOP", 0.0

        if self.backend == "neural" and self.neural_trainer is not None:
            return self.neural_trainer.predict_action(feature_vector)

        X = np.array([feature_vector], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        p_more = float(self.model.predict_proba(X_scaled)[0][1])
        action = "MORE" if p_more >= self.tau else "STOP"
        return action, round(p_more, 4)

    def save(self, filepath: str):
        """Save model and scaler to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.backend == "neural" and self.neural_trainer is not None:
            self.neural_trainer.save(filepath)
        else:
            with open(filepath, "wb") as f:
                joblib.dump({
                    "scaler": self.scaler,
                    "model": self.model,
                    "tau": self.tau,
                    "backend": "gbdt",
                    "is_trained": self.is_trained
                }, f)
            logger.info(f"Saved GBDT controller to {filepath}")

    def load(self, filepath: str):
        """Load model and scaler from disk with robust format detection."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        # Check if saved as a PyTorch checkpoint
        if filepath.endswith((".pt", ".pth")):
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch required to load .pt/.pth checkpoint.")
            self.backend = "neural"
            if self.neural_trainer is None:
                self.neural_trainer = PyTorchControllerTrainer(tau=self.tau)
            self.neural_trainer.load(filepath)
            self.scaler = self.neural_trainer.scaler
            self.tau = self.neural_trainer.tau
            self.model = self.neural_trainer.model
            self.is_trained = True
            return

        # Attempt joblib load
        with open(filepath, "rb") as f:
            data = joblib.load(f)

        if isinstance(data, dict):
            self.scaler = data.get("scaler", self.scaler)
            self.model = data.get("model", self.model)
            self.tau = data.get("tau", self.tau)
            self.backend = data.get("backend", "gbdt")
            self.is_trained = data.get("is_trained", True)
        else:
            # Fallback for bare estimators saved without dict packaging
            logger.warning("Loaded bare model estimator without dictionary metadata.")
            self.model = data
            self.is_trained = True
