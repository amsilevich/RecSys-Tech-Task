import numpy as np

from scipy import sparse
from tqdm import tqdm


class EvaluationMetrics:
    """
    Evaluates recommender system model quality
    """
    @staticmethod
    def mean_average_precision(real_recommendations: sparse.csr_matrix, predicted_recommendations: np.ndarray,
                               k: int = 10):
        """
        Calculates MAP@k

        Parameters
        ----------
        real_recommendations: sparse.csr_matrix
            real_recommendations[i, j] == 1 if j-th item recommends for i-th user
        predicted_recommendations: np.ndarray
            predicted_recommendations[i, j] == item_id if item_id is recommended for i-th user
        k: int = 10
            count

        Returns
        -------
            MAP@k: float
        """
        test_users_num = real_recommendations.shape[0]
        average_precisions = []
        for user in tqdm(range(test_users_num)):
            _, real_items = np.nonzero(real_recommendations[user])
            is_matched = np.isin(predicted_recommendations[user, :k], real_items)

            precisions_k = np.cumsum(is_matched) / (np.arange(len(is_matched)) + 1)
            average_precisions.append(precisions_k * is_matched)

        return np.mean(average_precisions).item()
