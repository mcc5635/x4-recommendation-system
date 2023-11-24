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