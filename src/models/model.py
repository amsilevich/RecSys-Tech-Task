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
        Fits the model on the interactions train data. Should be implemented in child classes.
        """
        pass

    def recommend(self, interactions: sparse.csr_matrix, target_users: np.ndarray, count: int) -> np.ndarray:
        """
        Recommends 'count' items for 'target_users' from interactions matrix. Should be implemented in child classes.
        """
        pass
