import numpy as np

from abc import ABC, abstractmethod
from scipy import sparse


class CollaborativeFilteringModel(ABC):
    """
    Model to recommend the items to user, using info about the part of user items and other users interactions.
    """

    @abstractmethod
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
        pass

    def recommend(self, target_users: np.ndarray, count: int) -> sparse.csr_matrix:
        """
        Recommends 'count' items for 'target_users'

        Parameters
        ----------
        target_users: np.ndarray
            Users ids which will receive the recommendations
        count: int
            Count of recommendations
        """
        pass