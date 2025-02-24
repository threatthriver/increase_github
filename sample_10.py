import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

class ModelError(Exception):
    """Base exception for model errors"""

class NotFittedError(ModelError):
    """Raised when prediction is attempted on an unfitted model"""

@dataclass
class BinaryClassifier:
    learning_rate: float = 0.01
    max_iterations: int = 1000
    weights: Optional[npt.NDArray] = None
    logger: logging.Logger = logging.getLogger("BinaryClassifier")

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> Tuple[float, float]:
        """Train the model using gradient descent."""
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(X.shape)}D")
        if len(y.shape) != 1:
            raise ValueError(f"Expected 1D array for labels, got {len(y.shape)}D")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        best_accuracy = 0.0
        best_loss = float("inf")

        for i in range(self.max_iterations):
            predictions = self._sigmoid(np.dot(X, self.weights))
            loss = self._compute_loss(y, predictions)
            accuracy = np.mean((predictions >= 0.5) == y)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_loss = loss

            gradient = np.dot(X.T, (predictions - y)) / n_samples
            self.weights -= self.learning_rate * gradient

            if i % 100 == 0:
                self.logger.info(
                    f"Iteration {i}: loss={loss:.4f}, accuracy={accuracy:.4f}"
                )

        return best_loss, best_accuracy

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict binary classes for X."""
        if self.weights is None:
            raise NotFittedError("Model must be fitted before prediction")
        return self._sigmoid(np.dot(X, self.weights)) >= 0.5

    @staticmethod
    def _sigmoid(x: npt.NDArray) -> npt.NDArray:
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    @staticmethod
    def _compute_loss(y: npt.NDArray, y_pred: npt.NDArray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))