import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
# turnLeft: 1000
# turnRight: 1000
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score


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

# 保存副本
# Turning left
LeftX = new_seq[:1000]
LeftY = [[1,0,0]] * 1000
# Turning right
RightX = new_seq[1000:2000]
RightY = [[0,0,1]] * 1000
# Going straight
StraightX = new_seq[2000:]
StraightY = [[0,1,0]] * 1000

# 创建y
y = [[1, 0, 0]] * 1000 + [[0, 0, 1]] * 1000 + [[0, 1, 0]] * 1000
# 同时打乱 记录随机数种子状态 -> 打乱new_seq -> 设置之前记录的随机数种子 -> 打乱y
state = np.random.get_state()
np.random.shuffle(new_seq)
np.random.set_state(state)
np.random.shuffle(y)

# 创建模型
model = Sequential()
# embedding
model.add(LSTM(128, input_shape=(40, 4)))
model.add(Dense(3, activation='softmax'))
model.summary()

# 转化为张量
X = tf.convert_to_tensor(new_seq, dtype=tf.float64)
Y = tf.convert_to_tensor(y, dtype=tf.float64)
print(X.shape)
print(Y.shape)


# 定义metrics
# 召回率
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# 精确率
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# F1
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
history = model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=10, shuffle=True)

left_predict = model.predict(np.asarray(LeftX).astype('float32'), batch_size= 10, verbose=1)
left_predict = (left_predict > 0.01).astype(int)
LeftY = np.reshape(LeftY, [-1])
left_predict = np.reshape(left_predict, [-1])
accuracy = accuracy_score(LeftY, left_predict)
precision = precision_score(LeftY, left_predict)
recall = recall_score(LeftY, left_predict)
f1score = f1_score(LeftY, left_predict)
print("left: ", "accuracy" , accuracy, " precision" , precision, " recall", recall, "f1score", f1score)

right_predict = model.predict(np.asarray(RightX).astype('float32'), batch_size= 10, verbose=1)
right_predict = (right_predict > 0.01).astype(int)
RightY = np.reshape(RightY, [-1])
right_predict = np.reshape(right_predict, [-1])
accuracy = accuracy_score(RightY, right_predict)
precision = precision_score(RightY, right_predict)
recall = recall_score(RightY, right_predict)
f1score = f1_score(RightY, right_predict)
print("right: ", "accuracy" , accuracy, " precision" , precision, " recall", recall, "f1score", f1score)

straight_predict = model.predict(np.asarray(StraightX).astype('float32'), batch_size= 10, verbose=1)
straight_predict = (straight_predict > 0.01).astype(int)
StraightY = np.reshape(StraightY, [-1])
straight_predict = np.reshape(straight_predict, [-1])
accuracy = accuracy_score(StraightY, straight_predict)
precision = precision_score(StraightY, straight_predict)
recall = recall_score(StraightY, straight_predict)
f1score = f1_score(StraightY, straight_predict)
print("straight: ", "accuracy" , accuracy, " precision" , precision, " recall", recall, "f1score", f1score)
# plot_model(model, to_file='model.png')
#
# # 绘制训练 & 验证的准确率值
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
