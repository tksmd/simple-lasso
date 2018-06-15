import numpy as np


def soft_threshold(X: np.ndarray, thresh: float):
    return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))


def coordinate_descent(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, n_iter: int = 1000) -> np.ndarray:
    n_samples = X.shape[0]
    n_features = X.shape[1]
    w = np.zeros(n_features)
    for _ in range(n_iter):
        for j in range(n_features):
            w[j] = 0.0
            r_j = y - np.dot(X, w)
            w[j] = soft_threshold(np.dot(X[:, j], r_j) / n_samples, alpha)
    return w
