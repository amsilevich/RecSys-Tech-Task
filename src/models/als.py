import numpy as np

from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm

from .model import CollaborativeFilteringModel


class ALSModel(CollaborativeFilteringModel):
    """
    Recommender model which works by ALS algorithm

    Parameters
    ----------
    rank: int
        Required rank of the approximation
    tolerance: float
        Stop when this distance between Im(U_k) and Im(U_{k+1}) is reached

    Attributes
    ----------
    rank: int
        Required rank of the approximation
    tolerance: float
        ALS algorithm ends when tolerance distance between Im(U_k) and Im(U_{k+1}) is reached
    VT: np.ndarray
        Matrix with orthogonal rows which contains the items embeddings
    """
    def __init__(self, rank: int = 30, tolerance: float = 1e-1):
        self.rank = rank
        self.tolerance = tolerance
        self.V = None

    @staticmethod
    def _evaluate_approximation(A, U, S, VT):
        """
        Calculates the Frobenius norm of difference matrix and it's approximation

        Parameters
        ----------
        A: sparse.csr_matrix
            Matrix to approximate.
        U, S, VT: np.ndarray
            Matrix approximation

        Returns
        -------
            ||A - USVT||_F: float
        """
        A_norm = sparse_norm(A)
        return np.sqrt(A_norm * A_norm
                       - np.trace(VT @ (A.T @ U) @ S)
                       - np.trace(S.T @ U.T @ A @ VT.T)
                       + np.trace(VT @ VT.T @ S.T @ U.T @ U @ S))

    @staticmethod
    def _dist_between_subspaces(U1: np.ndarray, U2: np.ndarray) -> float:
        """
        Parameters
        ----------
        U1, U2: np.ndarray
            Matrices with orthonormal columns

        Returns
        -------
           Distance between Im(U1) and Im(U2): float
        """
        U = np.concatenate([U1, -U2], 1)
        V = np.concatenate([U1, U2], 1)
        Qu, Ru = np.linalg.qr(U)
        Qv, Rv = np.linalg.qr(V)
        _, S, _ = np.linalg.svd(Ru @ Rv.T)
        return S[0]

    def fit(self, train_interactions: sparse.csr_matrix) -> 'CollaborativeFilteringModel':
        """
        Fits the model on the interactions train data

        Parameters
        ----------
        train_interactions: sparse.csr_matrix
            User-Item interactions

        Returns
        -------
            CollaborativeFilteringModel
                Fitted model
        """
        U_k, _ = np.linalg.qr(np.random.randn(train_interactions.shape[0], self.rank))
        distance = self._dist_between_subspaces(np.zeros(U_k.shape), U_k)
        while distance > self.tolerance:
            self.V, S_t = np.linalg.qr(train_interactions.T @ U_k)
            U_k_previous = U_k
            U_k, _ = np.linalg.qr(train_interactions @ self.V)
            distance = self._dist_between_subspaces(U_k_previous, U_k)
        return self

    def recommend(self, interactions: sparse.csr_matrix, target_users: np.ndarray, count: int) -> np.ndarray:
        """
        Recommends 'count' items for 'target_users'

        Parameters
        ----------
        interactions: sparse.csr_matrix
            User-Item interactions
        target_users: np.ndarray
            Users ids which will receive the recommendations
        count: int
            Count of recommendations

        Returns
        -------
            matrix: np.ndarray
            matrix[i, j] == item_id if item_id is recommended for i-th user from 'target_users'
        """
        recommendations = interactions[target_users] @ self.V @ self.V.T
        return np.argsort(recommendations, axis=1)[:, ::-1][:, :count]