import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random as random


df_train = pd.read_csv("glass.test", skiprows=1, header=None)
df_train[9]=LabelEncoder().fit_transform(df_train[9].values)
df_train[9].replace([0,1,2,3,4,5], [0,0,1,1,1,0],inplace=True)

df = pd.read_csv("glass.data", skiprows=1, header=None)
df[9]=LabelEncoder().fit_transform(df[9].values)
df[9].replace([0,1,2,3,4,5], [0,0,1,1,1,0],inplace=True)



def bagging(df_train,df_test, sampleTimes, trainTimes):
    result=pd.DataFrame(data=0,index=range(0,len(df_test)),columns=df_test[df_test.columns[-1]].unique())
    result_knn=pd.DataFrame(data=0,index=range(0,len(df_test)),columns=df_test[df_test.columns[-1]].unique())
    vote_result=[]
    vote_result_knn=[]
    for i in range(0, trainTimes):
        df_temp = df_train.iloc[0:1, :]
        # 随机采样        
        for j in range(0, sampleTimes):
            temp = random.randint(0, len(df_train)-1)
            df_temp = df_temp.append(df_train.loc[temp:temp], ignore_index=True)
        x_train=df_temp.iloc[:,:9]
        y_train=df_temp.iloc[:,9:]
        #创建弱训练器并训练
        clf = tree.DecisionTreeClassifier(random_state=42)
        knn = KNeighborsClassifier()
        clf.fit(x_train,y_train)
        knn.fit(x_train,y_train)
        x_test=df_test.iloc[:,:9]
        #获得单次训练器的结果
        result_temp=clf.predict(x_test)
        result_temp_knn=knn.predict(x_test)
        #存储结果用于投票
        count=0
        for item in result_temp:
            result[item][count]+=1
            count+=1
        count=0
        for item in result_temp_knn:
            result_knn[item][count]+=1
            count+=1
    #开始投票
    lists=list(result.columns)
    for i in range(0,len(df_test)):
        max=0
        temp=0
        for j in range(0,len(result.columns)):
            if max<result[lists[j]][i]:
                temp=j
                max=result[lists[j]][i]
        vote_result.append(lists[temp])
    for i in range(0,len(df_test)):
        max=0
        temp=0
        for j in range(0,len(result_knn.columns)):
            if max<result_knn[lists[j]][i]:
                temp=j
                max=result_knn[lists[j]][i]
        vote_result_knn.append(lists[temp])
    return vote_result,vote_result_knn

def score(test,result):
    success=0
    for i in range(0,len(result)):
        if(test[test.columns[-1]][i]==result[i]):
            success+=1
    return success/len(result)

#比较相同弱分类器数量时，数据集大小不同导致的结果差异
sizes = np.arange(int(len(df_train)/10),len(df_train)*3, int(len(df_train)/10))
result = {}
for size in sizes:
    result_tree,result_knn=bagging(df_train,df,size,50)
    score_tree     = score(df.iloc[:,9:],result_tree)
    score_knn     = score(df.iloc[:,9:],result_knn)
    result[size] = (score_tree,score_knn)
result = pd.DataFrame(result).transpose()
result.columns = ['Accuracy_tree','Accuracy_knn']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of training set')
plt.ylabel('Value')
plt.show()

#比较数据集大小相同时，弱分类器数量不同导致的结果差异比较数据集大小相同时，弱分类器数量不同导致的结果差异
sizes = np.arange(10,150,10)
result = {}
for size in sizes:
    result_tree,result_knn=bagging(df_train,df,140,size)
    score_tree     = score(df.iloc[:,9:],result_tree)
    score_knn     = score(df.iloc[:,9:],result_knn)
    result[size] = (score_tree,score_knn)
result = pd.DataFrame(result).transpose()
result.columns = ['Accuracy_tree','Accuracy_knn']
result.plot(marker='*', figsize=(15,5))
plt.xlabel('Size of classifier set')
plt.ylabel('Value')
plt.show()