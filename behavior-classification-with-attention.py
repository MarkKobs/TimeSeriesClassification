import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from attention import Attention
from keras.models import Sequential
import datetime

time_steps = 40
input_dim = 4
# 导入数据
turn_left_x = pd.read_csv('datasets/turnLeft_40.csv', header=0).values
turn_right_x = pd.read_csv('datasets/turnRight_40.csv', header=0).values
go_straight_x = pd.read_csv('datasets/goStraight_30.csv', header=0).values
turn_right_x = turn_right_x[:, 1:]
go_straight_x = go_straight_x[:, 1:]
# 将三者数据结合到一个数据结构list中
all_in_one = np.concatenate((turn_left_x, turn_right_x, go_straight_x), axis=0)
all_in_one = all_in_one[:, :4]
sequences = list()
# 数据切分，每次以长度40，划分为一个样本
for i in range(0, 2000):
    start = 40 * i
    end = start + 40
    sequences.append(all_in_one[start:end, :])
for i in range(2000, 3000):
    start = 30 * i
    end = start + 30

    sequences.append((all_in_one[start:end, :]))

# 数据补齐
to_pad = 40
new_seq = []
for one_seq in sequences:
    len_one_seq = len(one_seq)
    if len_one_seq != to_pad:
        n = to_pad - len_one_seq
        to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
        new_one_seq = np.concatenate([one_seq, to_concat])
    else:
        new_one_seq = one_seq
    new_seq.append(new_one_seq)

# 创建y
y = [[1, 0, 0]] * 1000 + [[0, 0, 1]] * 1000 + [[0, 1, 0]] * 1000
# 同时打乱 记录随机数种子状态 -> 打乱new_seq -> 设置之前记录的随机数种子 -> 打乱y
state = np.random.get_state()
np.random.shuffle(new_seq)
np.random.set_state(state)
np.random.shuffle(y)

# 转化为张量
X = tf.convert_to_tensor(new_seq, dtype=tf.float64)
Y = tf.convert_to_tensor(y, dtype=tf.float64)
print(X.shape)
print(Y.shape)

# Attention
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, input_dim)))
model.add(Attention(32))
model.add(Dense(3, activation="softmax"))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
starttime = datetime.datetime.now()
history = model.fit(X, Y, validation_split=0.33, epochs=10, batch_size=10, shuffle=True)
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
#
# # No Attention
# model2 = Sequential()
# model2.add(LSTM(128, return_sequences=True, input_shape=(time_steps, input_dim)))
# model2.add(LSTM(32))
# model2.add(Dense(3, activation='softmax'))
# model2.summary()
# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# starttime = datetime.datetime.now()
# history2 = model2.fit(X, Y, validation_split=0.33, epochs=10, batch_size=10, shuffle=True)
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
