"""
Model Diagnostics & Verification Suite.
Verifies tensor dimensions, forward/backward pass stability, gradient flow,
and numerical calibration before full training execution.
"""
import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from src.method_and_controller.router import ControllerRouter


class TestModelDiagnostics(unittest.TestCase):
    """Diagnostic verification tests for Adaptive Controller architecture."""

    def setUp(self):
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

    def test_sklearn_gbdt_forward_and_threshold(self):
        """Verify Scikit-learn GBDT backend dimensional alignment and probability calibration."""
        router = ControllerRouter(tau=0.3, backend="gbdt")
        
        # Synthetic 7-dim evidence features: [top1, mean, std, margin, range, max_sim, redundancy]
        X_mock = np.array([
            [0.90, 0.85, 0.02, 0.05, 0.10, 0.90, 0.80],
            [0.85, 0.80, 0.03, 0.04, 0.12, 0.85, 0.75],
            [0.55, 0.40, 0.15, 0.02, 0.35, 0.55, 0.20],
            [0.50, 0.35, 0.18, 0.01, 0.38, 0.50, 0.18],
        ], dtype=np.float32)
        y_mock = ["STOP", "STOP", "MORE", "MORE"]

        router.train(X_mock, y_mock)
        self.assertTrue(router.is_trained)

        # Inference on a single 7-dim feature vector
        sample_feat = [0.92, 0.87, 0.02, 0.05, 0.11, 0.92, 0.81]
        action, p_more = router.predict_action(sample_feat)

        self.assertIn(action, ["STOP", "MORE"])
        self.assertGreaterEqual(p_more, 0.0)
        self.assertLessEqual(p_more, 1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available in test environment")
    def test_pytorch_tensor_dimensions(self):
        """Verify input/hidden/output tensor shapes through the Residual MLP."""
        from src.method_and_controller.neural_router import NeuralEvidenceRouter

        batch_sizes = [1, 16, 64]
        in_features = 7
        hidden_dim = 64
        model = NeuralEvidenceRouter(in_features=in_features, hidden_dim=hidden_dim, num_res_blocks=2)
        model.eval()

        for b in batch_sizes:
            x = torch.randn(b, in_features)
            logits = model(x)
            probs = model.predict_proba(x)

            # Assert output dimensions
            self.assertEqual(logits.shape, (b, 1), f"Expected logits shape ({b}, 1), got {logits.shape}")
            self.assertEqual(probs.shape, (b, 1), f"Expected probs shape ({b}, 1), got {probs.shape}")

            # Assert probability bounds
            self.assertTrue(torch.all(probs >= 0.0).item())
            self.assertTrue(torch.all(probs <= 1.0).item())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available in test environment")
    def test_gradient_flow_and_backprop_stability(self):
        """
        Verify backward pass stability:
        1. All trainable parameters receive non-null gradients.
        2. No gradient norm is NaN or Inf.
        3. Gradient norms are bounded (no exploding or vanishing gradients).
        """
        from src.method_and_controller.neural_router import NeuralEvidenceRouter, FocalLoss

        model = NeuralEvidenceRouter(in_features=7, hidden_dim=64, num_res_blocks=2)
        model.train()
        loss_fn = FocalLoss(alpha=0.85, gamma=2.0)

        # Forward pass on synthetic batch
        x = torch.randn(32, 7, requires_grad=True)
        targets = torch.randint(0, 2, (32, 1)).float()

        logits = model(x)
        loss = loss_fn(logits, targets)

        # Assert finite loss
        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertFalse(torch.isinf(loss).item(), "Loss is Inf")
        self.assertGreater(loss.item(), 0.0, "Loss must be strictly positive")

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Audit gradient flow through every layer
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} did not receive a gradient!")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(param.grad).any().item(), f"Inf gradient in {name}")

                g_norm = param.grad.data.norm(2).item()
                grad_norms.append(g_norm)
                # Ensure no dead layers
                self.assertGreater(g_norm, 1e-9, f"Gradient vanished (norm={g_norm}) in layer: {name}")

        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        self.assertLess(avg_grad_norm, 20.0, f"Average gradient norm exploded: {avg_grad_norm}")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available in test environment")
    def test_focal_loss_class_imbalance_penalty(self):
        """
        Verify that FocalLoss correctly penalizes minority class false negatives
        more heavily than majority class false positives.
        """
        from src.method_and_controller.neural_router import FocalLoss

        loss_fn = FocalLoss(alpha=0.85, gamma=2.0)

        # Case A: False Negative on minority MORE class (target=1, predicted low logit)
        logit_fn = torch.tensor([[-2.0]])
        target_fn = torch.tensor([[1.0]])
        loss_minority_fn = loss_fn(logit_fn, target_fn).item()

        # Case B: False Positive on majority STOP class (target=0, predicted high logit)
        logit_fp = torch.tensor([[2.0]])
        target_fp = torch.tensor([[0.0]])
        loss_majority_fp = loss_fn(logit_fp, target_fp).item()

        # Alpha=0.85 should penalize minority miss more heavily than majority false positive
        self.assertGreater(
            loss_minority_fn,
            loss_majority_fp,
            "Focal loss must penalize minority (MORE) false negatives more heavily than majority false positives"
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available in test environment")
    def test_numerical_stability_extreme_inputs(self):
        """Verify model handles extreme input values without NaN or Inf explosion."""
        from src.method_and_controller.neural_router import NeuralEvidenceRouter

        model = NeuralEvidenceRouter(in_features=7, hidden_dim=64)
        model.eval()

        extreme_inputs = [
            torch.zeros(10, 7),              # All zeros
            torch.ones(10, 7) * 100.0,       # Large positive inputs
            torch.ones(10, 7) * -100.0,      # Large negative inputs
            torch.randn(10, 7) * 10.0        # High-variance inputs
        ]

        for x in extreme_inputs:
            logits = model(x)
            probs = model.predict_proba(x)

            self.assertFalse(torch.isnan(logits).any().item(), "NaN in logits on extreme inputs")
            self.assertFalse(torch.isinf(logits).any().item(), "Inf in logits on extreme inputs")
            self.assertFalse(torch.isnan(probs).any().item(), "NaN in probs on extreme inputs")


if __name__ == "__main__":
    unittest.main()
