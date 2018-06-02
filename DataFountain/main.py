# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler

import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn


def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r


start_all = datetime.datetime.now()
# path
# path_train = "/data/dm/train.csv"  # 训练文件路径
# path_test = "/data/dm/test.csv"  # 测试文件路径
# path_result_out = "model/pro_result.csv"
path_train = "dm/train.csv"  # 测试文件
path_test = "dm/newtest.csv"  # 测试文件
path_result_out = "model/pro_result1.csv"


# read train data
data = pd.read_csv(path_train)
train1 = []
alluser = data['TERMINALNO'].nunique()
# Feature Engineer, 对每一个用户生成特征:
# trip特征, record特征(数量,state等),
# 地理位置特征(location,海拔,经纬度等), 时间特征(星期,小时等), 驾驶行为特征(速度统计特征等)
for item in data['TERMINALNO'].unique():
    temp = data.loc[data['TERMINALNO'] == item, :]
    temp.index = range(len(temp))
    # trip 特征
    num_of_trips = temp['TRIP_ID'].nunique()
    trip_dis_list = []
    for item1 in temp['TRIP_ID'].unique():
        temp1 = temp.loc[temp['TRIP_ID'] == item1, :]
        temp1.index = range(len(temp1))
        startlong = temp1.loc[0, 'LONGITUDE']
        startlat = temp1.loc[0, 'LATITUDE']
        endlong = temp.iloc[-1, 3]
        endlat = temp.iloc[-1, 4]
        trip_dis = haversine1(startlong, startlat, endlong, endlat)
        trip_dis_list.append(trip_dis)
    trip_dis_df = pd.DataFrame(trip_dis_list, columns=["trip_dis"])
    mean_trip_dis = trip_dis_df["trip_dis"].mean()
    max_trip_dis = trip_dis_df["trip_dis"].max()

    # record 特征
    num_of_records = temp.shape[0]
    num_of_state = temp[['TERMINALNO', 'CALLSTATE']]
    nsh = num_of_state.shape[0]
    num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE']
        == 0].shape[0]/float(nsh)
    num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE']
        == 1].shape[0]/float(nsh)
    num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE']
        == 2].shape[0]/float(nsh)
    num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE']
        == 3].shape[0]/float(nsh)
    num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE']
        == 4].shape[0]/float(nsh)
    del num_of_state

    # 地点特征
    startlong = temp.loc[0, 'LONGITUDE']
    startlat = temp.loc[0, 'LATITUDE']
    hdis = haversine1(startlong, startlat, 113.9177317,
                      22.54334333)  # 距离某一点的距离
    # 时间特征
    temp['hour'] = temp['TIME'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).hour)
    hour_state = np.zeros([24, 1])
    for i in range(24):
        hour_state[i] = temp.loc[temp['hour'] == i].shape[0]/float(nsh)
    night_drive = hour_state[0]+hour_state[1] + \
        hour_state[2]+hour_state[3]+hour_state[4]
    busy_drive = hour_state[7]+hour_state[9]+hour_state[4]+hour_state[6]
    # 驾驶行为特征
    mean_speed = temp['SPEED'].mean()
    var_speed = temp['SPEED'].var()
    max_speed = temp['SPEED'].max()
    mean_height = temp['HEIGHT'].mean()
    mean_longitude = temp['LONGITUDE'].mean()
    mean_latitude = temp['LATITUDE'].mean()
    mean_0_speed = temp.loc[temp['CALLSTATE'] == 0]['SPEED'].mean()
    max_0_speed = temp.loc[temp['CALLSTATE'] == 0]['SPEED'].max()
    var_0_speed = temp.loc[temp['CALLSTATE'] == 0]['SPEED'].var()
    mean_1_speed = temp.loc[temp['CALLSTATE'] == 1]['SPEED'].mean()
    max_1_speed = temp.loc[temp['CALLSTATE'] == 1]['SPEED'].max()
    var_1_speed = temp.loc[temp['CALLSTATE'] == 1]['SPEED'].var()
    mean_2_speed = temp.loc[temp['CALLSTATE'] == 2]['SPEED'].mean()
    max_2_speed = temp.loc[temp['CALLSTATE'] == 2]['SPEED'].max()
    var_2_speed = temp.loc[temp['CALLSTATE'] == 2]['SPEED'].var()
    mean_3_speed = temp.loc[temp['CALLSTATE'] == 3]['SPEED'].mean()
    max_3_speed = temp.loc[temp['CALLSTATE'] == 3]['SPEED'].max()
    var_3_speed = temp.loc[temp['CALLSTATE'] == 3]['SPEED'].var()
    mean_4_speed = temp.loc[temp['CALLSTATE'] == 4]['SPEED'].mean()
    max_4_speed = temp.loc[temp['CALLSTATE'] == 4]['SPEED'].max()
    var_4_speed = temp.loc[temp['CALLSTATE'] == 4]['SPEED'].var()
    # 添加label
    target= temp.loc[0, 'Y']
    # 所有特征
    feature= [item, num_of_trips, mean_trip_dis, max_trip_dis, num_of_records,
               num_of_state_0, num_of_state_1, num_of_state_2, num_of_state_3, num_of_state_4,
               mean_speed, var_speed, max_speed, mean_height, mean_longitude, mean_latitude,
               mean_0_speed, max_0_speed, mean_1_speed, max_1_speed, mean_2_speed, max_2_speed, mean_3_speed, max_3_speed, mean_4_speed, max_4_speed,
               float(hour_state[0]), float(hour_state[1]), float(hour_state[2]), float(
                   hour_state[3]), float(hour_state[4]), float(hour_state[5]),
               float(hour_state[6]), float(hour_state[7]), float(hour_state[8]), float(
                   hour_state[9]), float(hour_state[10]), float(hour_state[11]),
               float(hour_state[12]), float(hour_state[13]), float(hour_state[14]), float(
                   hour_state[15]), float(hour_state[16]), float(hour_state[17]),
               float(hour_state[18]), float(hour_state[19]), float(hour_state[20]), float(
                   hour_state[21]), float(hour_state[22]), float(hour_state[23]),
               hdis,
               target]
    train1.append(feature)
train1= pd.DataFrame(train1)

# 特征命名
featurename= ['item', 'num_of_trips', 'mean_trip_dis', 'max_trip_dis', 'num_of_records',
               'num_of_state_0', 'num_of_state_1', 'num_of_state_2', 'num_of_state_3', 'num_of_state_4',
               'mean_speed', 'var_speed', 'max_speed', 'mean_height', 'mean_longitude', 'mean_latitude',
               'mean_0_speed', 'max_0_speed', 'mean_1_speed', 'max_1_speed', 'mean_2_speed', 'max_2_speed',
               'mean_3_speed', 'max_3_speed', 'mean_4_speed', 'max_4_speed',
               'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11',
               'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23',
               'dis',
               'target']
train1.columns= featurename

print("train data process time:", (datetime.datetime.now()-start_all).seconds)
# Train model
feature_use = ['item', 'num_of_trips', 'mean_trip_dis', 'max_trip_dis', 'num_of_records',
               'num_of_state_0', 'num_of_state_1', 'num_of_state_2', 'num_of_state_3', 'num_of_state_4',
               'mean_speed', 'var_speed', 'max_speed', 'mean_height', 'mean_longitude', 'mean_latitude',
               'mean_0_speed', 'max_0_speed', 'mean_1_speed', 'max_1_speed', 'mean_2_speed', 'max_2_speed',
               'mean_3_speed', 'max_3_speed', 'mean_4_speed', 'max_4_speed',
               'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11',
               'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23',
               'dis']

feature_use1 = ['num_of_trips', 'mean_trip_dis', 'max_trip_dis', 'num_of_records',
                'num_of_state_0', 'num_of_state_1', 'num_of_state_2', 'num_of_state_3', 'num_of_state_4',
                'mean_speed', 'var_speed', 'max_speed', 'mean_height', 'mean_longitude', 'mean_latitude',
                'mean_0_speed', 'max_0_speed', 'mean_1_speed', 'max_1_speed', 'mean_2_speed', 'max_2_speed',
                'mean_3_speed', 'max_3_speed', 'mean_4_speed', 'max_4_speed'
                'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11',
                'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23',
                'dis']
# The same process for the test set
data= pd.read_csv(path_test)
test1= []
for item in data['TERMINALNO'].unique():
    temp = data.loc[data['TERMINALNO'] == item, :]
    temp.index= range(len(temp))
    # trip 特征
    num_of_trips= temp['TRIP_ID'].nunique()
    trip_dis_list = []
    for item1 in temp['TRIP_ID'].unique():
        temp1= temp.loc[temp['TRIP_ID'] == item1, :]
        temp1.index= range(len(temp1))
        startlong= temp1.loc[0, 'LONGITUDE']
        startlat= temp1.loc[0, 'LATITUDE']
        endlong= temp.iloc[-1, 3]
        endlat= temp.iloc[-1, 4]
        trip_dis = haversine1(startlong, startlat, endlong, endlat)
        trip_dis_list.append(trip_dis)
    trip_dis_df= pd.DataFrame(trip_dis_list, columns=["trip_dis"])
    mean_trip_dis = trip_dis_df["trip_dis"].mean()
    max_trip_dis = trip_dis_df["trip_dis"].max()

    # record 特征
    num_of_records= temp.shape[0]
    num_of_state = temp[['TERMINALNO', 'CALLSTATE']]
    nsh= num_of_state.shape[0]
    num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE'] == 0].shape[0]/float(nsh)
    num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE'] == 1].shape[0]/float(nsh)
    num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE'] == 2].shape[0]/float(nsh)
    num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE'] == 3].shape[0]/float(nsh)
    num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE'] == 4].shape[0]/float(nsh)
    del num_of_state
    # 地点特征
    startlong= temp.loc[0, 'LONGITUDE']
    startlat= temp.loc[0, 'LATITUDE']
    hdis = haversine1(startlong, startlat, 113.9177317, 22.54334333)
    # 时间特征
    temp['hour']= temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    hour_state=np.zeros([24, 1])
    for i in range(24):
        hour_state[i]=temp.loc[temp['hour'] == i]['hour'].shape[0]/float(nsh)
    # 驾驶行为特征
    mean_speed=temp['SPEED'].mean()
    var_speed=temp['SPEED'].var()
    max_speed=temp['SPEED'].max()
    mean_height=temp['HEIGHT'].mean()
    mean_longitude=temp['LONGITUDE'].mean()
    mean_latitude=temp['LATITUDE'].mean()
    mean_0_speed=temp.loc[temp['CALLSTATE'] == 0]['SPEED'].mean()
    max_0_speed=temp.loc[temp['CALLSTATE'] == 0]['SPEED'].max()
    mean_1_speed=temp.loc[temp['CALLSTATE'] == 1]['SPEED'].mean()
    max_1_speed=temp.loc[temp['CALLSTATE'] == 1]['SPEED'].max()
    mean_2_speed=temp.loc[temp['CALLSTATE'] == 2]['SPEED'].mean()
    max_2_speed=temp.loc[temp['CALLSTATE'] == 2]['SPEED'].max()
    mean_3_speed=temp.loc[temp['CALLSTATE'] == 3]['SPEED'].mean()
    max_3_speed=temp.loc[temp['CALLSTATE'] == 3]['SPEED'].max()
    mean_4_speed=temp.loc[temp['CALLSTATE'] == 4]['SPEED'].mean()
    max_4_speed=temp.loc[temp['CALLSTATE'] == 4]['SPEED'].max()
    # test标签设为-1
    target=-1.0
    feature=[item, num_of_trips, mean_trip_dis, max_trip_dis, num_of_records,
               num_of_state_0, num_of_state_1, num_of_state_2, num_of_state_3, num_of_state_4,
               mean_speed, var_speed, max_speed, mean_height, mean_longitude, mean_latitude,
               mean_0_speed, max_0_speed, mean_1_speed, max_1_speed, mean_2_speed, max_2_speed, mean_3_speed, max_3_speed, mean_4_speed, max_4_speed,
               float(hour_state[0]), float(hour_state[1]), float(hour_state[2]), float(
                   hour_state[3]), float(hour_state[4]), float(hour_state[5]),
               float(hour_state[6]), float(hour_state[7]), float(hour_state[8]), float(
                   hour_state[9]), float(hour_state[10]), float(hour_state[11]),
               float(hour_state[12]), float(hour_state[13]), float(hour_state[14]), float(
                   hour_state[15]), float(hour_state[16]), float(hour_state[17]),
               float(hour_state[18]), float(hour_state[19]), float(hour_state[20]), float(
                   hour_state[21]), float(hour_state[22]), float(hour_state[23]),
               hdis,
               target]
    test1.append(feature)
test1=pd.DataFrame(test1)
test1.columns=featurename
print("test data process time:", (datetime.datetime.now()-start_all).seconds)


# 去掉id特征
X_train=train1[feature_use].iloc[:, 1:].fillna(-1)
y_train=train1['target']
X_test=test1[feature_use].iloc[:, 1:].fillna(-1)
# 标准化
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)

# 采用lgb模型
model_lgb=lgb.LGBMRegressor(objective='regression', num_leaves=13,
                              learning_rate=0.01, n_estimators=680,
                              max_depth=13, boosting_type='gbdt',
                              max_bin=82, bagging_fraction=0.5,
                              bagging_freq=5, feature_fraction=0.3534,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_child_weight=0.0001, min_child_samples=20, subsample=1.0,
                              subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                              reg_lambda=0.0, silent=True,
                              min_data_in_leaf=7, min_sum_hessian_in_leaf=14)

model=model_lgb
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("train success")


# output result
result=pd.DataFrame(test1['item'])
result['pre']=y_pred
result=result.rename(columns={'item': 'Id', 'pre': 'Pred'})
result.to_csv(path_result_out, header=True, index=False)
print("Time used:", (datetime.datetime.now()-start_all).seconds)
# '''
