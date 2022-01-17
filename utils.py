import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    sigmoid function for arrays
    """
    s = 1 / (1 + np.exp(-z))

    return s


def initialize_parameters(
    dim: int, method: str = "zeros", random_state: int = 330
) -> np.ndarray:
    """
    initialize network parameters using "method"
    method in ("zeros", "random")
    """
    if method == "zeros":
        w = np.zeros((dim, 1))
    elif method == "random":
        w = np.random.RandomState(random_state).randn(dim, 1)
    else:
        raise ValueError(
            f"method can only be 'zeros' or 'random' but found {method}"
        )

    return w


def concatenate_ones(X: np.ndarray) -> np.ndarray:
    """
    concatenate one to each feature vector

    args:
        X: input data of shape (num_features/n_x x num_samples/m)

    return:
        np.ndarray of shape ((n_x + 1) x m)
    """
    ones = np.ones((1, X.shape[1]))
    X = np.concatenate((ones, X), axis=0)

    return X
