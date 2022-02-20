# -*- coding:utf-8 -*-
# @Time : 2022/2/19 11:30 下午
# @Author : huichuan LI
# @File : pop.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from collections import Counter
from model.abstract_model import GeneralRecommender
from dataset import Dataset


class Pop(GeneralRecommender):
    """
    Pop is an fundamental model that always recommend the most popular item.
    """

    def __init__(self, config, dataset):
        self.k = config['k']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.build_model(dataset.merge)

    def build_model(self, data):
        self.item_cnt = Counter(data['anime_id'])
        self.item_id = [elem[0] for elem in self.item_cnt.most_common(self.k)]
        self.max_cnt = self.item_cnt.most_common(1)[0][1]
        return 0

    def calculate_loss(self, data):
        return 0

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        result = [elem//self.max_cnt for elem in [self.item_cnt[v] for v in item]]
        return result

    def full_sort_predict(self, interaction):
        batch_user_num = interaction[self.USER_ID].shape[0]
        result = {key: (value / self.max_cnt) for key, value in self.item_cnt.items()}
        return result.view(-1)


if __name__ == "__main__":
    config = {'dataset': 'anime_data',
              'USER_ID_FIELD': "user_id", "ITEM_ID_FIELD": "anime_id", "LABEL_FIELD": "rating", "TIME_FIELD": "",
              "interaction_path": "/Users/hui/Desktop/python/recommendation_algo/data/rating.csv", "k": 10,
              "item_path": "/Users/hui/Desktop/python/recommendation_algo/data/parsed_anime.csv", "user_path": ""}
    dataset = Dataset(config=config)
    pop = Pop(config, dataset=dataset)
    print(pop.predict(dataset.merge.iloc[:3, ]))
