import numpy as np


class PCA:
    """
    自实现 PCA 类，用于降维和重建
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X: np.ndarray):
        # 去中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # 协方差矩阵
        cov = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
        # SVD 分解
        U, S, _ = np.linalg.svd(cov)
        # 选取前 n_components
        self.components_ = U[:, : self.n_components]
        self.explained_variance_ = S[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        X_rec = np.dot(Z, self.components_.T) + self.mean_
        return X_rec

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
