import numpy as np

from scipy import sparse
from tqdm import tqdm

from .interactions import InteractionsDataset


class TrainTestSplitter:
    """
    Class which performs split the data on train and test sets
    """

    @staticmethod
    def split_interactions(filtered_interactions: InteractionsDataset, items_for_user_threshold: int,
                           test_items_for_user: int, test_data_percent: float = 0.2) -> tuple[sparse.csr_matrix,
                                                                                              sparse.csr_matrix,
                                                                                              np.ndarray]:
        """
        Splits the 'filtered_interactions' dataset

        Parameters
        ----------
        filtered_interactions: Interactions
            User-Item interactions
        items_for_user_threshold: int
            Minimum count of items for user to be a candidate in the test set
        test_items_for_user: int
            Count of items which will be in test set
        test_data_percent: float, from 0 to 1
            Percent of the data to be in test set

        Returns
        -------
            (interactions_train, interactions_test, test_users): tuple[sparse.csr_matrix, sparse.csr_matrix,
                                                                       np.ndarray[int]]
        """
        csr_interactions = filtered_interactions.get_features_sparse_matrix()
        potential_test_users, _ = np.nonzero((csr_interactions.sum(axis=1) > items_for_user_threshold))
        users_num = csr_interactions.shape[0]
        test_users = np.random.choice(potential_test_users, int(test_data_percent * users_num))

        interactions_train: sparse.csr_matrix = csr_interactions.copy()
        interactions_test: sparse.csr_matrix = csr_interactions[test_users].copy()
        for test_user in tqdm(test_users):
            _, user_items = np.nonzero(csr_interactions[test_user])
            test_items = np.random.choice(user_items, test_items_for_user)
            interactions_train[test_user, test_items] = 0

        interactions_test = csr_interactions[test_users] - interactions_train[test_users]

        return interactions_train, interactions_test, test_users
