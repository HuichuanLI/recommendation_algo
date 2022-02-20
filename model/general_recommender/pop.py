# -*- coding:utf-8 -*-
# @Time : 2022/2/19 11:30 下午
# @Author : huichuan LI
# @File : pop.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from collections import Counter


class Pop():
    """
    Pop is an fundamental model that always recommend the most popular item.
    """

    def __init__(self, k):
        self.k = k

    def build_model(self, data):
        self.item_cnt = Counter(data['anime_id'])
        self.item_id = [elem[0] for elem in self.item_cnt.most_common(self.k)]
        return 0

    def calculate_loss(self, data):
        return 0

    def predict(self, test):
        n = len(test)
        return [self.item_id for _ in range(n)]


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv("../../data/rating.csv")

    data = data[data["rating"] > 4]
    pop = Pop(10)
    print(pop.build_model(data))
    print(pop.predict(data))
