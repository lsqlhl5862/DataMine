# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import random as random

df = pd.read_csv('Transactions.csv')


class merchandise:
    name = ""
    sum = 0
    buy = 0
    average = 0
    variance = 0

    def __init__(self, n):
        self.name = n
        
    def computeAverage(self):
        self.average = self.buy/self.sum

    def computeVariance(self):
        for i in range(self.buy):
            self.variance += (1-self.average)**2
        for i in range(self.sum-self.buy):
            self.variance += self.average**2
        self.variance /= self.sum

    def display(self):
        print(self.name, self.buy, self.average, self.variance)


temp = list(df.columns)
lists = []
for name in temp:
    lists.append(merchandise(name))

temp = df.sum()
lenth = df.shape[0]
sum = 0
for i in range(df.shape[1]):
    lists[i].buy = temp[i]
    sum += temp[i]
    lists[i].sum = lenth
    lists[i].computeAverage()
    lists[i].computeVariance()

print('商品名', '购买数', '均值', '方差')
for i in range(len(lists)):
    lists[i].display()

print('购买总数：', sum)

df.replace(0, -1, inplace=True)

# 插入并修改列
df['wine'] = -1
ran = random.sample(range(0, len(df)), len(df)//5)
for i in ran:
    df['wine'][i] = 1


# 插入并修改行
df.loc[len(df)] = -1
ran = random.sample(range(0, df.shape[1]), df.shape[1]//4)
lists.append(merchandise('wine'))
for i in ran:
    df[lists[i].name][df.shape[0]-1] = 1

# 写入文件
df.to_csv("Results.csv")
