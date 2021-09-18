'''
PINNs的pytorch实现
基本方法一致，不同的问题仅需在类中更换不同的网络、待求参数、优化算法以及损失函数
'''

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import math
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"    #若需要使用CPU训练，去掉此行首部注释符号'#'即可
print("Training device:", device)

EPOCH = 1000
BATCH_SIZE = 300

a = 1.0
b = 100.0
l = 1.0

#定义神经网络类
class PINNs(nn.Module):
    def __init__(self):
        #初始化pytorch父类
        super().__init__()
        #定义网络各层
        self.model = nn.Sequential(
            nn.Linear(2, 3000),
            nn.Tanh(),
            nn.Linear(3000, 3000),
            nn.Tanh(),
            nn.Linear(3000, 1)
        )
        #定义优化器
        self.lr = 0.5
        self.optimiser = torch.optim.LBFGS(self.parameters(), lr=self.lr)

    def forward(self, inputs):
        return self.model(inputs)

    def loss_func(self, inputs, outputs):
        self.u_t_x = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        self.u_tt_xx = torch.autograd.grad(outputs=self.u_t_x, inputs=inputs, grad_outputs=torch.ones_like(self.u_t_x), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        self.temp_1 = torch.tensor([[1.0], [0.0]], requires_grad=True).to(device)
        self.temp_2 = torch.tensor([[0.0], [1.0]], requires_grad=True).to(device)
        self.f = torch.mm(self.u_tt_xx, self.temp_1)-torch.mm(self.u_tt_xx, self.temp_2)-(b*torch.sinh(torch.mm(inputs, self.temp_2)))
        self.temp_3 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], requires_grad=True).to(device)
        self.temp_4 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], requires_grad=True).to(device)
        self.u_xtozero = torch.mm(inputs, self.temp_3)
        self.u_xtol = torch.mm(inputs, self.temp_3)+torch.mm(torch.full([BATCH_SIZE, 2], l, requires_grad=True).to(device), self.temp_4)
        self.u_ttozero = torch.mm(inputs, self.temp_4)
        self.u_t_x_ttozero = torch.autograd.grad(outputs=self.model(self.u_ttozero), inputs=self.u_ttozero, grad_outputs=torch.ones_like(self.model(self.u_ttozero)), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        self.MSE = (self.f).pow(2)+(self.model(self.u_xtozero)).pow(2)+(self.model(self.u_xtol)).pow(2)+(self.model(self.u_ttozero)).pow(2)+(torch.mm(self.u_t_x_ttozero, self.temp_1)).pow(2)
        return self.MSE

    def train(self, inputs):
        def closure():
            outputs = self.forward(inputs)
            loss = self.loss_func(inputs, outputs).mean()
            print("Current loss:", loss)
            self.optimiser.zero_grad()
            loss.backward(retain_graph=True)
            return loss
        self.optimiser.step(closure)
        self.optimiser = torch.optim.LBFGS(self.parameters(), lr=self.lr)

#生成训练集
temptrainlist=[]
    #内部取样点
for i in range(0, 6000):
    t = random.uniform(0, 2)
    x = random.uniform(0, l)
    temptrainlist.append([t, x])
    #三侧边缘随机取样点
for i in range(0, 1000):
    t = random.uniform(0, 2)
    temptrainlist.append([t, 0])
for i in range(0, 1000):
    t = random.uniform(0, 2)
    temptrainlist.append([t, l])
for i in range(0, 1000):
    x = random.uniform(0, l)
    temptrainlist.append([0, x])
trainlist = torch.tensor(temptrainlist, requires_grad=True)
train_loader = Data.DataLoader(dataset=trainlist, batch_size=BATCH_SIZE, shuffle=True)

#开始训练
PINNsModel = PINNs().to(device)
for epoch in range(1, EPOCH+1):
    for i, data in enumerate(train_loader, 0):
        torch.cuda.empty_cache()
        data = data.to(device)
        PINNsModel.train(data)
        print("Batch", i, "has been used in training")
    if((epoch%50) == 0):
        PINNsModel.lr *= 0.9
    print("Epoch", epoch, "has finished, and the current learning rate is", PINNsModel.lr)
torch.save(PINNsModel, "./wave equation net.pkl")

#编写目标函数解析式及绘图
def u(t, x):
    temp = 0.0
    for n in range(1, 1001):
        temp += (((-1)**(n+1))*(math.sin((n*(math.pi)*x)/l))*(1-math.cos((a*n*(math.pi)*t)/l)))/(n*((n**2)*((math.pi)**2)+(l**2)))
    return ((2.0*b*(l**2))*(math.sinh(l))*temp)/((a**2)*(math.pi))
x_slice = 100
y_slice = 100
t = np.linspace(0, 2, x_slice, dtype=np.float32)
x = np.linspace(0, 1, y_slice, dtype=np.float32)
T, X = np.meshgrid(t, x)
Z = np.array(np.arange(0, 1, (1.0/(x_slice*y_slice)))).reshape(x_slice, y_slice)
for i in range(0, x_slice):
    for j in range(0, y_slice):
        Z[i][j] = u(T[i][j], X[i][j])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, Z, cmap=cm.YlGnBu_r)
plt.show()

#绘制拟合函数图像并创建测试集对比训练结果与目标函数的差异
x_slice = 100
y_slice = 100
t = np.linspace(0, 2, x_slice, dtype=np.float32)
x = np.linspace(0, 1, y_slice, dtype=np.float32)
T, X = np.meshgrid(t, x)
Z = np.array(np.arange(0, 1, (1.0/(x_slice*y_slice)))).reshape(x_slice, y_slice)
for i in range(0, x_slice):
    for j in range(0, y_slice):
        temp = torch.tensor([T[i][j], X[i][j]], requires_grad=True).to(device)
        Z[i][j] = PINNsModel.forward(temp).tolist()[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, Z, cmap=cm.YlGnBu_r)
plt.show()
error = []
for i in range(1, 10001):
    t = random.uniform(0, 2)
    x = random.uniform(0, l)
    temp = torch.tensor([t, x], requires_grad=True).to(device)
    error.append((PINNsModel.forward(temp).tolist()[0]-u(t, x))**2)
print("Mean squared error of the model:", np.mean(error))