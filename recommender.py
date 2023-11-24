from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisiticMatrixFactorization
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from typing import Dict, Any


MODEL = {
    "lmf": LogisticMatrixFactorization,
    "als": AlternatingLeastSquares,
    "bpr": BayesianPersoanlizedRanking,
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
        self.user_product_matrix = _get_sparse_matrix
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
            # Balance positive and negative (nan) entries
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