import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_pre import data
from numpy import shape
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, random_split, TensorDataset
import random

# 生成n个1*4的随机数据
# n = 100  # 可以根据需要修改
# data = []  # 用来存储数据的列表
# start = random.randint(10, 100)  # 生成一个随机的起始值
# for i in range(n):
#     # 生成一个1*4的随机数据，其中前三维是随机数，第四维是自然数
#     row = [random.uniform(0, 10) for _ in range(3)]  # 前三维是0到1之间的随机数，可以根据需要修改范围
#     row.append(i + start)  # 第四维是自然数，从?开始
#     data.append(row)  # 将数据添加到列表中

# 打印数据
# for row in data:
#     print(row)
# 定义时间窗口大小，即每次输入多少个点来预测下一个点
window_size = 10
n = len(data)


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=640, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=640, hidden_size=100, batch_first=True)
        self.linear1 = nn.Linear(in_features=100, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=30)
        self.linear3 = nn.Linear(in_features=30, out_features=60)
        self.linear4 = nn.Linear(in_features=60, out_features=3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # select the last time step of each batch
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)

        x = self.linear3(x)
        x = torch.relu(x)

        x = self.linear4(x)
        return x


model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
# 训练模型，使用32个批次大小，进行100个周期，每10个周期打印一次训练结果
batch_size = 32
epochs = 10
batch_num = math.ceil(n / batch_size)


def time_label(a: float, b: float, precision=0.1):
    if (b - a) / precision < window_size:
        print("时间差过小")
        exit(1)
    # noinspection SpellCheckingInspection
    timelabel = []
    # print("a, b=", a, b)
    a1 = math.ceil(a / precision - 10)
    b1 = math.ceil(b / precision - (window_size - 1))
    # print("a1, b1= ", a1, b1)
    for i in range(a1, b1):
        temp = []
        for j in range(window_size):
            temp.append((i + j) * precision)
        timelabel.append(temp)
    return timelabel


dataset = []
for i in range(batch_num):
    temp = []
    for j in range(batch_size):
        if i * batch_size + j == n:
            break
        temp.append(data[i * batch_size + j])
    dataset.append(temp)

input_dataset = []
for batch in dataset:
    label_start = batch[0][3]
    label_last = batch[-1][3]
    label = time_label(label_start, label_last)
    label = np.array(label)
    label = label.reshape(len(label), window_size, 1)
    temp_label = torch.from_numpy(label)
    temp_label = temp_label.float()

    data_temp = np.array(batch)
    temp_data = data_temp[:, :3]
    temp_data = torch.from_numpy(temp_data)
    temp_data = temp_data.float()
    input_dataset.append([temp_label, temp_data])
epoch_loss = 0
for epoch in range(epochs):
    for state in input_dataset:
        input_label, train_data = state
        result = model(input_label)
        if epoch == 0:
            print("每轮input_label", input_label)
            print("每轮result", result)
            print("每轮train_data", train_data)
        loss = criterion(result, train_data)
        # 反向传播，计算梯度
        loss.backward(retain_graph=True)
        # 更新参数，进行梯度下降
        optimizer.step()

        # 清零梯度，避免累积
        optimizer.zero_grad()

        # 累加损失值到变量中
        epoch_loss += loss.item()
        # if (epoch + 1) % 10 == 0:
        #     print(loss.item())

# result = model(input_dataset[0][0])
result1 = torch.empty(0, 3)

for state in input_dataset:
    _label, _data = state
    _result = model(_label)
    result1 = torch.cat((result1, _result.detach()), dim=0)
result1 = result1.numpy()
print(shape(result1))
print(result1)
data = np.array(data)
arr1, arr2, arr3, arr4 = np.split(data, 4, axis=1)  # 分割成3个子数组，沿着列轴
brr1, brr2, brr3 = np.split(result1, 3, axis=1)  # 分割成3个子数组，沿着列轴
fig = plt.figure()  # 创建一个图形对象
ax = fig.add_subplot(projection="3d")  # 创建一个带有3D投影的子图对象
# 绘制第一个曲线
ax.plot(arr1, arr2, arr3, label="data", color="blue")

# 绘制第二个曲线
ax.plot(brr1, brr2, brr3, label="trained", color="red")

# 设置标题，图例和坐标轴标签
ax.set_title("train and data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

# 显示或保存图形
plt.show()
# plt.savefig("3d_curve.png")
