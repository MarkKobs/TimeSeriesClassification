# TimeSeriesClassification
**LSTM, classification, vehicle behavior recognition, attention mechanism**

- 导入数据，数据是由scenic定义并由carlo生成的十字路口仿真数据
- 需要将生成的数据处理成神经网络可以读取的数据格式
    1. 将三者的数据合并到一个python的数据结构list中
    2. 数据切分，每次以长度40进行切分，因为样本最长设置为40
    3. 数据补齐，对于长度不足40的样本，根据最后一行的内容进行复制，延长至40
- 打标签标签
```java
以one-hot的方式对label进行编码
[1,0,0]表示左转
[0,0,1]表示右转
[0,1,0]表示直行
此前设置的每种类型有1000个样本，因此每个长度为1000，总共3000
```

- 由于x和y之前是把三种车辆行为的数据简单连接，需要进行打乱, 而x和y其实有一一对应关系，所以需要同时打乱
    1. 记录随机数种子状态 
    2. 打乱x
    3. 设置当前状态为之前记录的随机数种子
    4. 打乱y
- 创建模型
    1. input_shape设置为40,5 
    2. 40表示一个样本的数量 5表示特征维度  64表示隐藏节点数量
    3. 加一层softmax，将LSTM输出的64维度的向量转换为维度为3的向量
- 将x和y都转成tensorflow框架所支持的张量形式
```python
X = tf.convert_to_tensor(new_seq, dtype=tf.float64)
Y = tf.convert_to_tensor(y, dtype=tf.float64)
```
- 自定义各种metrics
1. recall
2. precision
3. f1
4. 还有自带的accuracy
- 编译模型
    1. model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
    2. 由于是多分类问题，所以选择交叉熵损失函数，优化器选择adam
选择输出四类评价指标

