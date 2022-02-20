# -*- coding:utf-8 -*-
# @Time : 2022/2/20 2:11 下午
# @Author : huichuan LI
# @File : dataset.py
# @Software: PyCharm
import numpy as np
from scipy.sparse import coo_matrix
from logging import getLogger
import pandas as pd


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._get_field_from_config()
        self.load_data()
        self.inter_matrix()

    def load_data(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.rating = pd.read_csv(self.config['interaction_path'])
        if self.config['item_path']:
            self.item_feature = pd.read_csv(self.config['item_path'])
            self.merge = pd.merge(self.rating, self.item_feature, on=self.iid_field)
        if self.config['user_path']:
            self.user_deature = pd.read_csv(self.config['user_path'])
            self.merge = pd.merge(self.merge, self.user_deature, on=self.uid_field)
        return self.merge

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                'USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time.'
            )

        # self.logger.debug(set_color('uid_field', 'blue') + f': {self.uid_field}')
        # self.logger.debug(set_color('iid_field', 'blue') + f': {self.iid_field}')

    def inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        return self._create_sparse_matrix(self.merge, self.uid_field, self.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.
        Source and target should be token-like fields.
        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.
        Args:
            df_feat (Interaction): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.
        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `df_feat`\'s features.')
            data = df_feat[value_field]
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))
        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

        return mat

    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.
        Args:
            field (str): field name to get token number.
        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        return np.max(self.merge[field]) + 1


if __name__ == "__main__":
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "NEG_PREFIX": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv",
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": ""}
    dataset = Dataset(config=config)
    print(dataset.merge)
