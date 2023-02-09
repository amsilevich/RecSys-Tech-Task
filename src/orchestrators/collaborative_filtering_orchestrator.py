import typing as tp
import numpy as np

from scipy import sparse

from .orchestrator import Orchestrator
from src.columns import Columns
from src.datasets.interactions import InteractionsDataset
from src.datasets.train_test_splitter import TrainTestSplitter
from src.evaluator import EvaluationMetrics
from src.models.als import ALSModel
from src.models.model import CollaborativeFilteringModel


class CFOrchestrator(Orchestrator):
    def __init__(self, config: dict[str, tp.Any], interactions: InteractionsDataset) -> None:
        self.config = config
        self.model: tp.Optional[CollaborativeFilteringModel] = None
        self.interactions = interactions

    def _prepare_data(self) -> None:
        data_config = self.config.get('data')
        self.interactions = self.interactions.filter_non_informative_data(Columns.UserInteractions,
                                                                          data_config.get('users_threshold'))\
                                             .filter_non_informative_data(Columns.ItemInteractions,
                                                                          data_config.get('items_threshold'))

    def _split_data(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
        splitter_config = self.config.get('splitter')
        splitter = TrainTestSplitter()
        return splitter.split_interactions(
            self.interactions,
            splitter_config.get('items_for_user_threshold'),
            splitter_config.get('test_items_for_user'),
            splitter_config.get('test_data_percent')
        )

    def _fit_model(self, interactions_train: sparse.csr_matrix) -> None:
        model_config = self.config.get('model')
        if model_config.get('name') == 'ALSModel':
            self.model = ALSModel(model_config.get('rank'), model_config.get('tolerance'))
        self.model.fit(interactions_train)

    def _recommend(self, interactions_train: sparse.csr_matrix, test_users: np.ndarray) -> np.ndarray:
        recommend_config = self.config.get('recommend')
        return self.model.recommend(interactions_train, test_users, recommend_config.get('count'))

    def _evaluate_result(self, interactions_test: sparse.csr_matrix, recommendations: np.ndarray) -> float:
        evaluate_config = self.config.get('metric')
        if evaluate_config.get('name') == 'mean_average_precision':
            metric = EvaluationMetrics().mean_average_precision
            return metric(interactions_test, recommendations, evaluate_config.get('count'))

    def run(self) -> float:
        self._prepare_data()
        interactions_train, interactions_test, test_users = self._split_data()
        self._fit_model(interactions_train)
        recommendations = self._recommend(interactions_train, test_users)
        quality_metric = self._evaluate_result(interactions_test, recommendations)
        return quality_metric