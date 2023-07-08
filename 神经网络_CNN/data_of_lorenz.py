#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


def lorenz(state, t):
    # 解包状态变量
    x, y, z = state

    # 定义洛伦兹方程的参数
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # 计算洛伦兹方程的导数
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z

    return [x_dot, y_dot, z_dot]


# In[3]:


# 定义初始状态和时间范围
state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 40.0, 0.01)

# In[4]:


# 使用odeint求解洛伦兹方程组
states = odeint(lorenz, state0, t)

# In[6]:


# 绘制结果图形并显示出来。
# plt.savefig('lorenz.jpg')
# plt.savefig('lorenz.pdf') 图片保存
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.show()

# In[12]:


# print(states[:, 0])

# In[8]:


# print(states[1])

# In[23]:

#
# for i in range(0, 4000, 10):
#     print(np.append(states[i], (0.01 * i)))


