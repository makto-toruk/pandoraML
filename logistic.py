"""
a basic implementation of logistic regression
for binary classification
from a neural network perspective
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .utils import concatenate_ones, initialize_parameters, sigmoid


class LogisticRegression:
    def __init__(
        self,
        eta: float = 0.001,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        verbose: bool = False,
    ) -> None:

        self.max_iter = max_iter
        self.eta = eta
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        return None

    def propagate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:

        A = sigmoid(np.dot(self.w.T, X))
        cost = (
            -1
            / self.m
            * (np.dot(y, np.log(A).T) + np.dot(1 - y, np.log(1 - A).T))
        )
        cost = np.squeeze(np.array(cost))

        grads = 1 / self.m * np.dot(X, (A - y).T)

        return grads, cost

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LogisticRegression:
        """
        optimize parametes to reduce loss function

        args:
            X: input data of shape (num_features/n_x x num_samples/m)
        """
        # TODO: implementation for sample_weight is missing
        if self.fit_intercept:
            X = concatenate_ones(X)

        self.m = X.shape[1]
        self.w = initialize_parameters(X.shape[0])

        self.costs = []
        for i in range(self.max_iter):
            grads, cost = self.propagate(X, y)
            self.w -= self.eta * grads
            self.costs.append(cost)
            if self.verbose:
                if i % 100 == 0:
                    print(f"Cost after iteration {i} = {cost}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        predict class probabilities
        """
        if self.fit_intercept:
            X = concatenate_ones(X)

        return sigmoid(np.dot(self.w.T, X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict class labels
        """
        proba = self.predict_proba(X)
        y_pred = (proba > 0.5).astype(int)

        return y_pred

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        # TODO: implementation for sample weight is missing
        y_pred = self.predict(X)
        accuracy = 100 - np.mean(np.abs(y_pred - y)) * 100

        return accuracy
