import pandas as pd

from collections import Counter
from scipy import sparse

from src.columns import Columns


class InteractionsDataset:
    """
    Class to contain users-items interactions

    Parameters
    ----------
    interactions: pd.DataFrame
        Table of interactions which contains columns:
            - 'Columns.UserInteractions' - user_id
            - 'Columns.ItemInteractions' - item_id
            - 'Columns.DataInteractions' - result of interaction (by default = 1)

    Attributes
    ----------
    interactions: pd.DataFrame
        Table of interactions

    Raises
    ------
        KeyError: if there isn't come necessary columns in interactions table
    """

    @staticmethod
    def _check_columns_correctness(interactions: pd.DataFrame) -> None:
        """
        Checks the columns names correctness in interactions.

        Parameters
        ----------
        interactions: pd.DataFrame
            Table of interactions

        Returns
        -------
            None

        Raises
        ------
            KeyError: if there isn't come necessary columns in interactions table
        """
        custom_columns = {Columns.UserInteractions, Columns.UserInteractions, Columns.DataInteractions}
        if not custom_columns.issubset(interactions.columns):
            raise KeyError(f"User-Item interactions table should contain at least columns: "
                           f"{Columns.UserInteractions}, {Columns.UserInteractions}, {Columns.DataInteractions}")

    def __init__(self, interactions: pd.DataFrame) -> None:
        self._check_columns_correctness(interactions)
        self.interactions = interactions

    def filter_non_informative_data(self, column: str, threshold: int) -> 'InteractionsDataset':
        """
        Filters rows of the data which contain less than `threshold` values in `column` column

        Parameters
        ----------
        column: str
            Name of the column to filter by
        threshold: str
            Threshold to filter by

        Returns
        -------
            InteractionsDataset
        """
        counter = Counter(self.interactions[column].values)
        self.interactions = \
            self.interactions[self.interactions.apply(lambda row: counter[row['row']] >= threshold, axis=1)]
        return self

    def to_matrix(self) -> sparse.csr_matrix:
        """
        Converts interactions table to compressed sparse matrix

        Returns
        -------
            sparse.csr_matrix
        """
        matrix = sparse.csr_matrix(
            (
                self.interactions[Columns.DataInteractions].values,
                 (
                     self.interactions[Columns.UserInteractions].values,
                     self.interactions[Columns.ItemInteractions].values
                 )
            )
        )
        return matrix
