import numpy as np
import pandas as pd

from scipy import sparse

from src.columns import Columns
from src.datasets.dataset import Dataset


class UsersDataset(Dataset):
    """
    Class to contain users features

    Parameters
    ----------
    user_age: pd.DataFrame
        Table of users ages which contains columns:
            - 'Columns.User' - user id
            - 'Columns.UserAge' - user age
    user_region: pd.DataFrame
        Table of users regions which contains columns:
            - 'Columns.User' - user id
            - 'Columns.UserRegion' - user region (one-hot encoded format)

    Attributes
    ----------
    user_age: pd.DataFrame
        Table of users ages
    user_region: pd.DataFrame
        Table of users regions

    Raises
    ------
        KeyError: if there isn't come necessary columns in init tables
    """
    def __init__(self, user_age: pd.DataFrame, user_region: pd.DataFrame) -> None:
        user_age_columns = {Columns.User, Columns.UserAge}
        user_region_columns = {Columns.User, Columns.UserRegion}
        self._check_columns_correctness((user_age_columns, user_age), (user_region_columns, user_region))

        self.user_age = user_age
        self.user_region = user_region

    def get_features_sparse_matrix(self) -> sparse.csr_matrix:
        """
        Combines all tables with user info and stores to compressed sparse matrix.
        i-th row of the matrix responds to the user's by i-th id features.

        Returns
        -------
            sparse.csr_matrix
        """
        users_num = max(self.user_age.shape[0], self.user_region.shape[0])

        user_region_crs = sparse.csr_matrix(
            (
                np.ones(self.user_age.shape[0]),
                (
                    self.user_age[Columns.User],
                    self.user_age[Columns.UserAge]
                )
            ),
            shape=(users_num, self.user_age[Columns.UserAge].max() + 1)
        )

        user_age_csr = sparse.csr_matrix(
            (
                self.user_age[Columns.UserAge].values,
                (
                    self.user_age[Columns.User].values,
                    np.zeros(self.user_age.shape[0], dtype=int)
                )
            ),
            shape=(users_num, 1)
        )

        return sparse.hstack([user_region_crs, user_age_csr])
