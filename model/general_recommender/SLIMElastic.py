# -*- coding:utf-8 -*-
# @Time : 2022/2/27 4:21 下午
# @Author : huichuan LI
# @File : SLIMElastic.py
# @Software: PyCharm

from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
from model.abstract_model import GeneralRecommender
import scipy.sparse as sp
import warnings
import numpy as np
from dataset import Dataset


class SLIMElastic(GeneralRecommender):
    r"""SLIMElastic is a sparse linear method for top-K recommendation, which learns
    a sparse aggregation coefficient matrix by solving an L1-norm and L2-norm
    regularized optimization problem.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.hide_item = config['hide_item']
        self.alpha = config['alpha']
        self.l1_ratio = config['l1_ratio']
        self.positive_only = config['positive_only']

        X = dataset.inter_matrix(form='csr').astype(np.float32)
        X = X.tolil()
        self.interaction_matrix = X
        print(X.shape)
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            positive=self.positive_only,
            fit_intercept=False,
            copy_X=False,
            precompute=True,
            selection='random',
            max_iter=100,
            tol=1e-4
        )
        item_coeffs = []

        # ignore ConvergenceWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            for j in range(X.shape[1]):
                # target column
                r = X[:, j]

                if self.hide_item:
                    # set item column to 0
                    X[:, j] = 0

                # fit the model
                model.fit(X, r.todense().getA1())

                # store the coefficients
                coeffs = model.sparse_coef_

                item_coeffs.append(coeffs)

                if self.hide_item:
                    # reattach column if removed
                    X[:, j] = r

        self.item_similarity = sp.vstack(item_coeffs).T
        self.other_parameter_name = ['interaction_matrix', 'item_similarity']

    def forward(self):
        pass

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        r = (self.interaction_matrix[user, :].multiply(self.item_similarity[:, item].T)).sum(axis=1).getA1()

        return r

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        r = self.interaction_matrix[user, :] @ self.item_similarity
        r = r.todense().getA1()

        return r


if __name__ == "__main__":
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "hide_item": True,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": "",
              "alpha": 0.2, "l1_ratio": 0.02, "positive_only": True}
    print("load_data")
    dataset = Dataset(config=config)
    print("training")
    knn = SLIMElastic(config, dataset=dataset)
    print(knn.predict(dataset.merge.iloc[:3, ]))
