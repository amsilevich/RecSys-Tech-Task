import pandas as pd

from abc import ABC, abstractmethod
from scipy import sparse


class Dataset(ABC):
    """
    Abstract dataset class
    """
    @staticmethod
    def _check_columns_correctness(*args: tuple[set[str], pd.Dataframe]):
        """
        Checks the columns names correctness in input dataframes.

        Parameters
        ----------
        args: tuple[set[str], pd.Dataframe]
            Each arg is pair of custom_columns and columns from input table.

        Returns
        -------
            None

        Raises
        ------
            KeyError: if there isn't some necessary columns in init tables
        """
        for custom_columns, table in args:
            if not custom_columns.issubset(table.columns):
                raise KeyError(f"Table should contain at least columns: "
                               f"{', '.join(custom_columns)}")

    @abstractmethod
    def get_features_sparse_matrix(self) -> sparse.csr_matrix:
        """
        Combines all tables with data and stores to compressed sparse matrix.
        i-th row of the matrix responds to the user's by i-th id features.

        Returns
        -------
            sparse.csr_matrix
        """
        pass
