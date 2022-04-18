import numpy as np
import pandas as pd
import csv

np.random.seed(25)

# data = pd.read_csv(r"E:\网课\推荐系统\作业\hw1\ml-1m(1)\ml-1m\ratings.dat",header=None,encoding="utf-8",delimiter="::",quoting=csv.QUOTE_NONE)

data_path = '/home/liushaofan/RS_Learning/data/'

# 数据集载入
A = []
with open(data_path+'ml-1m/ratings.dat','r') as f:
    for line in f:
        userID,MovieID,rating,step = line.split('::')
        # print(step)
        A.append([int(userID)-1,int(MovieID)-1,int(rating)])

# for i in A:
#     print(i)

user_num = 6040
movie_num = 3952

# 划分训练集数据集并处理
index_ = range(len(A))
train_index = np.random.choice(a=index_, size=int(len(A) * 0.8), replace=False)
test_index = list(set(index_) - set(train_index))

print(len(index_), len(train_index), len(test_index))

train_data_ = np.zeros(shape=(user_num, movie_num))
test_data_ = np.zeros(shape=(user_num, movie_num))

for i in train_index:
    train_data_[A[i][0]][A[i][1]] = A[i][2]

for i in test_index:
    test_data_[A[i][0]][A[i][1]] = A[i][2]

# 对于没有评分的电影，使用已有评分的电影的平均分来初始化
train_mean = np.sum(train_data_, axis=1, keepdims=True) / (np.sum(1-np.equal(train_data_, 0), axis=1, keepdims=True) + 1e-6)
test_mean = np.sum(test_data_, axis=1, keepdims=True) / (np.sum(1-np.equal(test_data_, 0), axis=1, keepdims=True) + 1e-6)

train_data = np.where(np.equal(train_data_, 0), np.tile(train_mean, (1, np.shape(train_data_)[1])), train_data_)
test_data = np.where(np.equal(test_data_, 0), np.tile(test_mean, (1, np.shape(test_data_)[1])), test_data_)

# 计算相似度矩阵
def sim_mat(rating, mean):
    # mean = np.mean(rating, axis=1)
    # A = rating - np.tile(mean, (1, np.shape(rating)[1]))
    A = rating
    fenzi = np.matmul(A, A.T)
    B = np.sqrt(np.sum(np.square(A),axis=-1,keepdims=True))
    fenmu = np.matmul(B, B.T)
    sim = fenzi / (fenmu + 1e-6)
    return sim

# 计算预测矩阵 .
def pred(sim, N=None):
    ind = np.argsort(sim)[:, -N:]  # 最高相似度用户下标

    pred_rating = np.zeros_like(train_data)

    for i in range(len(ind)):
        pred_rating[i,:] = np.mean(train_data[ind[i]], axis=0)

    return pred_rating


# 计算预测误差
sim = sim_mat(train_data, train_mean)
print("sim[0] ", sim[0])

pred_test = pred(sim, 5)
MSE = np.mean((test_data - pred_test)**2)
print("Test_data MSE error: ", MSE)

pred_train = pred(sim, 5)
MSE = np.mean((train_data - pred_test)**2)
# MSE = np.sum(np.where(1 - np.equal(train_data_, 0), (train_data_ - pred_train)**2, 0)) / (len(train_index))
print("Train_data MSE error: ", MSE)