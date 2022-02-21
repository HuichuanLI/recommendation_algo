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
        self.user_num = np.max(self.rating[self.uid_field]) + 1
        self.item_num = np.max(self.rating[self.iid_field]) + 1

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

    def history_item_matrix(self, value_field=None):
        """Get dense matrix describe user's history interaction records.
        ``history_matrix[i]`` represents user ``i``'s history interacted item_id.
        ``history_value[i]`` represents user ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.
        ``history_len[i]`` represents number of user ``i``'s history interaction records.
        ``0`` is used as padding.
        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(row='user', value_field=value_field)

    def history_user_matrix(self, value_field=None):
        """Get dense matrix describe item's history interaction records.
        ``history_matrix[i]`` represents item ``i``'s history interacted item_id.
        ``history_value[i]`` represents item ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.
        ``history_len[i]`` represents number of item ``i``'s history interaction records.
        ``0`` is used as padding.
        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(row='item', value_field=value_field)

    def _history_matrix(self, row, value_field=None):
        """Get dense matrix describe user/item's history interaction records.
        ``history_matrix[i]`` represents ``i``'s history interacted item_id.
        ``history_value[i]`` represents ``i``'s history interaction records' values.
            ``0`` if ``value_field = None``.
        ``history_len[i]`` represents number of ``i``'s history interaction records.
        ``0`` is used as padding.
        Args:
            row (str): ``user`` or ``item``.
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.
        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """

        user_ids, item_ids = self.rating[self.uid_field].to_list(), self.rating[self.iid_field].to_list()
        if value_field is None:
            values = np.ones(len(self.merge))
        else:
            if value_field not in self.rating.columns:
                raise ValueError(f'Value_field [{value_field}] should be one of `inter_feat`\'s features.')
            values = self.rating[value_field].to_list()

        if row == 'user':
            row_num, max_col_num = self.user_num, self.item_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.item_num, self.user_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)
        if col_num > max_col_num * 0.2:
            self.logger.warning(
                f'Max value of {row}\'s history interaction records has reached '
                f'{col_num / max_col_num * 100}% of the total.'
            )

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return history_matrix, history_value, history_len


if __name__ == "__main__":
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "NEG_PREFIX": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv",
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": ""}
    dataset = Dataset(config=config)
    print(dataset.merge)
    print(dataset.history_user_matrix(value_field="rating_x"))
