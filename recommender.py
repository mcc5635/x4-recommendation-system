from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from typing import Dict, Any

MODEL = {
    "lmf": LogisticMatrixFactorization,
    "als": AlternatingLeastSquares,
    "bpr": BayesianPersonalizedRanking,
}


def _get_sparse_matrix(values, user_idx, product_idx):
    return csr_matrix(
        (values, (user_idx, product_idx)),
        shape=(len(user_idx.unique()), len(product_idx.unique())),
    )


def _get_model(name: str, **params):
    model = MODEL.get(name)
    if model is None:
        raise ValueError("No model with name {}".format(name))
    return model(**params)


class InternalStatusError(Exception):
    pass


class Recommender:
    def __init__(
        self,
        values,
        user_idx,
        product_idx,
    ):
        self.user_product_matrix = _get_sparse_matrix(values, user_idx, product_idx)
        self.user_idx = user_idx
        self.product_idx = product_idx

        # This variable will be set during training phase
        self.model = None
        self.fitted = False

    def create_and_fit(
        self,
        model_name: str,
        weight_strategy: str = "bm25",
        model_params: Dict[str, Any] = {},
    ):
        weight_strategy = weight_strategy.lower()
        if weight_strategy == "bm25":
            data = bm25_weight(
                self.user_product_matrix,
                K1=1.2,
                B=0.75,
            )
        elif weight_strategy == "balanced":
            # Balance the positive and negative (nan) entries
            # http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
            total_size = (
                self.user_product_matrix.shape[0] * self.user_product_matrix.shape[1]
            )
            sum = self.user_product_matrix.sum()
            num_zeros = total_size - self.user_product_matrix.count_nonzero()
            data = self.user_product_matrix.multiply(num_zeros / sum)
        elif weight_strategy == "same":
            data = self.user_product_matrix
        else:
            raise ValueError("Weight strategy not supported")

        self.model = _get_model(model_name, **model_params)
        self.fitted = True

        self.model.fit(data)

        return self

    def recommend_products(
        self,
        user_id,
        items_to_recommend=5,
    ):
        """Finds the recommended items for the user.

        Returns:
            (items, scores) pair, where item is already the name of the suggested item.
        """

        if not self.fitted:
            raise InternalStatusError(
                "Cannot recommend products without previously fitting the model."
                " Please, consider fitting the model before recommening products."
            )

        return self.model.recommend(
            user_id,
            self.user_product_matrix[user_id],
            filter_already_liked_items=True,
            N=items_to_recommend,
        )

    def explain_recommendation(
        self,
        user_id,
        suggested_item_id,
        recommended_items,
    ):
        _, items_score_contrib, _ = self.model.explain(
            user_id,
            self.user_product_matrix,
            suggested_item_id,
            N=recommended_items,
        )

        return items_score_contrib

    def similar_users(self, user_id):
        return self.model.similar_users(user_id)

    @property
    def item_factors(self):
        return self.model.item_factors