import numpy as np
import pandas as pd

from scipy import sparse

from src.columns import Columns
from src.datasets.dataset import Dataset


class ItemsDataset(Dataset):
    """
    Class to contain items features

    Parameters
    ----------
    item_asset: pd.DataFrame
        Table of items assets which contains columns:
            - 'Columns.Item' - item id
            - 'Columns.ItemAsset' - item asset
    item_price: pd.DataFrame
        Table of items prices which contains columns:
            - 'Columns.Item' - item id
            - 'Columns.ItemPrice' - item price
    item_subclass: pd.DataFrame
        Table of items subclasses which contains columns:
            - 'Columns.Item' - item id
            - 'Columns.ItemSubclass' - item subclass

    Attributes
    ----------
    item_asset: pd.DataFrame
        Table of items assets
    item_price: pd.DataFrame
        Table of items prices
    item_subclass: pd.DataFrame
        Table of items subclasses

    Raises
    ------
        KeyError: if there isn't come necessary columns in init tables
    """
    def __init__(self, item_asset: pd.DataFrame, item_price: pd.DataFrame, item_subclass: pd.DataFrame) -> None:
        item_asset_columns = {Columns.Item, Columns.ItemAsset}
        item_price_columns = {Columns.Item, Columns.ItemPrice}
        item_subclass_columns = {Columns.Item, Columns.ItemSubclass}
        self._check_columns_correctness(
            (item_asset_columns, item_asset),
            (item_price_columns, item_price),
            (item_subclass_columns, item_subclass),
        )

        self.item_asset, self.item_price, self.item_subclass = item_asset, item_price, item_subclass

    def get_features_sparse_matrix(self) -> sparse.csr_matrix:
        """
        Combines all tables with items info and stores to compressed sparse matrix.
        i-th row of the matrix responds to the item's by i-th id features.

        Returns
        -------
            sparse.csr_matrix
        """
        items_num = max(self.item_asset.shape[0], self.item_price.shape[0], self.item_subclass.shape[0])

        item_asset_csr = sparse.csr_matrix(
            (
                self.item_asset[Columns.ItemAsset].values,
                (
                    self.item_asset[Columns.Item].values,
                    np.zeros(self.item_asset.shape[0], dtype=int)
                )
            ),
            shape=(items_num, 1)
        )

        item_price_csr = sparse.csr_matrix(
            (
                self.item_price[Columns.ItemPrice].values,
                (
                    self.item_price[Columns.Item].values,
                    np.zeros(self.item_price.shape[0], dtype=int)
                )
            ),
            shape=(items_num, 1)
        )

        item_subclass_csr = sparse.csr_matrix(
            (
                np.ones(self.item_subclass.shape[0]),
                (
                    self.item_subclass[Columns.Item],
                    self.item_subclass[Columns.ItemSubclass]
                )
            ),
            shape=(items_num, self.item_subclass[Columns.ItemSubclass].max() + 1)
        )

        return sparse.hstack([item_asset_csr, item_price_csr, item_subclass_csr])

