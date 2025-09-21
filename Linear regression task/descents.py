from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type
import numpy as np
from scipy.sparse import issparse


@dataclass
class LearningRate:
    """Class for calculating learning rate with adaptive formula."""
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5
    iteration: int = 0

    def __call__(self) -> float:
        """
        Calculates learning rate using formula: lambda_ * (s0 / (s0 + t))^p.

        :return: Current learning rate (float)
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """Enumeration of available loss functions."""
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """Base class for gradient descent methods."""

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Initialization of base gradient descent class.

        :param dimension: Feature space dimension
        :param lambda_: Learning rate parameter
        :param loss_function: Optimized loss function
        """
        self.w: np.ndarray = np.random.randn(dimension) * 0.01  # Initialize small random weights
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def _ensure_dense(self, x: np.ndarray) -> np.ndarray:
        """
        Converts sparse matrix to dense if needed.

        :param x: Feature matrix (sparse or dense)
        :return: Dense matrix
        """
        return x.toarray() if issparse(x) else x

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs one gradient descent step.

        :param x: Feature matrix
        :param y: Target vector
        :return: Weight difference (w_{k+1} - w_k)
        """
        gradient = self.calc_gradient(x, y)
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            raise ValueError("Gradient contains NaN or Inf values")
        return self.update_weights(gradient)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates weights based on gradient (template).

        :param gradient: Loss function gradient
        :return: Weight difference (w_{k+1} - w_k)
        """
        raise NotImplementedError("update_weights method not implemented")

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates gradient of loss function with respect to weights.

        :param x: Feature matrix
        :param y: Target vector
        :return: Gradient
        """
        x_dense = self._ensure_dense(x)
        n_samples = x.shape[0]
        y_pred = self.predict(x)
        residuals = y_pred - y

        if self.loss_function == LossFunction.MSE:
            if issparse(x):
                gradient = (2 / n_samples) * (x.T @ residuals)
            else:
                gradient = (2 / n_samples) * x_dense.T @ residuals
        elif self.loss_function == LossFunction.LogCosh:
            if issparse(x):
                gradient = (1 / n_samples) * (x.T @ np.tanh(residuals))
            else:
                gradient = (1 / n_samples) * x_dense.T @ np.tanh(residuals)
        else:
            raise NotImplementedError(f"Gradient not implemented for {self.loss_function}")

        return gradient

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates loss for given features and target vector.

        :param x: Feature matrix
        :param y: Target vector
        :return: Loss value
        """
        y_pred = self.predict(x)
        if self.loss_function == LossFunction.MSE:
            return np.mean((y - y_pred) ** 2)
        elif self.loss_function == LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(y_pred - y)))
        else:
            raise NotImplementedError(f"Loss not implemented for {self.loss_function}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts target values for given features.

        :param x: Feature matrix
        :return: Predicted values
        """
        return x @ self.w if issparse(x) else self._ensure_dense(x) @ self.w


class VanillaGradientDescent(BaseDescent):
    """Full gradient descent class."""

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates weights based on gradient.

        :param gradient: Loss function gradient
        :return: Weight difference (w_{k+1} - w_k)
        """
        step_size = self.lr()
        gradient = np.clip(gradient, -1e6, 1e6)  # Gradient clipping
        w_diff = -gradient * step_size
        self.w += w_diff
        return w_diff


class StochasticDescent(VanillaGradientDescent):
    """Stochastic gradient descent class."""

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: Feature space dimension
        :param lambda_: Learning rate parameter
        :param batch_size: Batch size
        :param loss_function: Optimized loss function
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates gradient based on random batch.

        :param x: Feature matrix
        :param y: Target vector
        :return: Gradient
        """
        n_samples = x.shape[0]
        actual_batch_size = min(self.batch_size, n_samples)
        batch_indices = np.random.choice(n_samples, size=actual_batch_size, replace=False)

        x_batch = x[batch_indices] if issparse(x) else x[batch_indices]
        y_batch = y[batch_indices]

        y_pred = self.predict(x_batch)
        residuals = y_pred - y_batch

        if self.loss_function == LossFunction.MSE:
            gradient = (2 / actual_batch_size) * (x_batch.T @ residuals)
        elif self.loss_function == LossFunction.LogCosh:
            gradient = (1 / actual_batch_size) * (x_batch.T @ np.tanh(residuals))
        else:
            raise NotImplementedError(f"Gradient not implemented for {self.loss_function}")

        return gradient


class MomentumDescent(VanillaGradientDescent):
    """Gradient descent with momentum class."""

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: Feature space dimension
        :param lambda_: Learning rate parameter
        :param loss_function: Optimized loss function
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9
        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates weights with momentum.

        :param gradient: Loss function gradient
        :return: Weight difference (w_{k+1} - w_k)
        """
        step_size = self.lr()
        gradient = np.clip(gradient, -1e6, 1e6)
        self.h = self.alpha * self.h + step_size * gradient
        w_diff = -self.h
        self.w += w_diff
        return w_diff


class Adam(VanillaGradientDescent):
    """Adaptive Moment Estimation (Adam) gradient descent class."""

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: Feature space dimension
        :param lambda_: Learning rate parameter
        :param loss_function: Optimized loss function
        """
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8
        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.beta_1: float = 0.9
        self.beta_2: float = 0.999
        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates weights using Adam algorithm.

        :param gradient: Loss function gradient
        :return: Weight difference (w_{k+1} - w_k)
        """
        self.iteration += 1
        step_size = self.lr()
        gradient = np.clip(gradient, -1e6, 1e6)

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)

        w_diff = -(step_size / (np.sqrt(v_hat) + self.eps)) * m_hat
        self.w += w_diff
        return w_diff


class BaseDescentReg(BaseDescent):
    """Base class with L2 regularization."""

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: L2 regularization coefficient
        """
        super().__init__(*args, **kwargs)
        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates gradient of loss function with L2 regularization.

        :param x: Feature matrix
        :param y: Target vector
        :return: Gradient
        """
        l2_gradient = 2 * self.mu * self.w
        return super().calc_gradient(x, y) + l2_gradient


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """Full gradient descent with L2 regularization."""
    pass


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """Stochastic gradient descent with L2 regularization."""
    pass


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """Gradient descent with momentum and L2 regularization."""
    pass


class AdamReg(BaseDescentReg, Adam):
    """Adam algorithm with L2 regularization."""
    pass


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Returns gradient descent class instance based on configuration.

    :param descent_config: Configuration dictionary (descent_name, regularized, kwargs)
    :return: Gradient descent class instance
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f"Incorrect descent method name, use one of: {descent_mapping.keys()}")

    descent_class = descent_mapping[descent_name]
    return descent_class(**descent_config.get('kwargs', {}))