from __future__ import annotations
from typing import List
import numpy as np
from scipy.sparse import issparse
from descents import BaseDescent, get_descent, LossFunction

class LinearRegression:
    """
    Linear regression class with support for sparse matrices and various loss functions.
    """
    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        Initialization of linear regression model.

        :param descent_config: Descent method configuration (descent_name, regularized, kwargs)
        :param tolerance: Tolerance for stopping criterion (weight difference norm)
        :param max_iter: Maximum number of iterations
        :param loss_function: Loss function (MSE, LogCosh, etc.)
        """
        self.descent: BaseDescent = get_descent({
            **descent_config,
            "kwargs": {**descent_config.get("kwargs", {}), "loss_function": loss_function}
        })
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Train the model on data.

        :param x: Feature matrix (sparse or dense)
        :param y: Target vector
        :return: Trained model
        """
        # Handle sparse matrices
        if issparse(x):
            n_features = x.shape[1]
            x_dense = x.toarray()  # Convert to dense format for training
        else:
            n_features = x.shape[1]
            x_dense = x

        # Initialize weights if they are not defined or have wrong dimension
        if self.descent.w is None or self.descent.w.shape[0] != n_features:
            self.descent.w = np.random.randn(n_features) * 0.01

        # Initial loss
        self.loss_history.append(self.calc_loss(x, y))

        for i in range(self.max_iter):
            try:
                w_old = np.copy(self.descent.w)

                # Perform gradient descent step
                weight_diff = self.descent.step(x_dense, y)

                # Calculate current loss
                loss = self.calc_loss(x, y)
                self.loss_history.append(loss)

                # Check for numerical issues
                if np.isnan(loss) or np.isinf(loss):
                    # print(f"Stopping at iteration {i}: loss instability (loss = {loss})")
                    self.descent.w = w_old
                    break

                if np.any(np.isnan(self.descent.w)) or np.any(np.isinf(self.descent.w)):
                    # print(f"Stopping at iteration {i}: NaN/Inf in weights")
                    self.descent.w = w_old
                    break

                # Check for convergence
                weight_change_norm = np.linalg.norm(weight_diff) ** 2
                if weight_change_norm < self.tolerance:
                    # print(f"Convergence reached at iteration {i}: weight change norm = {weight_change_norm:.2e}")
                    break

            except Exception as e:
                # print(f"Stopping at iteration {i}: {type(e).__name__} - {e}")
                self.descent.w = w_old
                break

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict for given features.

        :param x: Feature matrix (sparse or dense)
        :return: Predicted values
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for given features and target vector.

        :param x: Feature matrix (sparse or dense)
        :param y: Target vector
        :return: Loss value
        """
        return self.descent.calc_loss(x, y)