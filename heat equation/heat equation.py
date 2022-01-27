'''
PINNs的pytorch实现
基本方法一致，不同的问题仅需在类中更换不同的网络、待求参数、优化算法以及损失函数
'''

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import integrate
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

EPOCH = 50
BATCH_SIZE = 100

a = 0.3
l = (3*(math.pi))/4.0
h = 1.0

#定义神经网络类
class PINNs(nn.Module):
    def __init__(self):
        #初始化pytorch父类
        super().__init__()
        #定义网络各层
        self.model = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )
        #定义求解过程中需要优化的参数
        mylambda = torch.rand((1, 1), requires_grad=True)
        self.mylambda = nn.Parameter(mylambda)
        #定义优化器
        self.lr = 0.05
        self.optimiser = torch.optim.LBFGS(self.parameters(), lr=self.lr)

    def forward(self, inputs):
        return self.model(inputs)

    def loss_func(self, inputs, outputs, labels):
        self.u_t_x = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        self.u_tt_xx = torch.autograd.grad(outputs=self.u_t_x, inputs=inputs, grad_outputs=torch.ones_like(self.u_t_x), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        self.temp_1 = torch.tensor([[1.0], [0.0]], requires_grad=True).to(device)
        self.temp_2 = torch.tensor([[0.0], [1.0]], requires_grad=True).to(device)
        self.f = torch.mm(self.u_t_x, self.temp_1)-((self.mylambda[0])**2)*torch.mm(self.u_tt_xx, self.temp_2)
        self.temp_3 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], requires_grad=True).to(device)
        self.temp_4 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], requires_grad=True).to(device)
        self.u_xtozero = torch.mm(inputs, self.temp_3)
        self.u_xtol = torch.mm(inputs, self.temp_3)+torch.mm(torch.full([BATCH_SIZE, 2], l, requires_grad=True).to(device), self.temp_4)
        self.u_ttozero = torch.mm(inputs, self.temp_4)
        self.u_t_x_xtol = torch.autograd.grad(outputs=self.model(self.u_xtol), inputs=self.u_xtol, grad_outputs=torch.ones_like(self.model(self.u_xtol)), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)[0]
        self.MSE = (self.f).pow(2)+((self.model(self.u_ttozero))-torch.sin(torch.mm(inputs, self.temp_2))).pow(2)+(self.model(self.u_xtozero)).pow(2)+(torch.mm(self.u_t_x_xtol, self.temp_2)+h*(self.model(self.u_xtol))).pow(2)+torch.pow((self.model(inputs)-labels), 2)
        return self.MSE

    def train(self, inputs):
        def closure():
            input = torch.mm(inputs, torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], requires_grad=True).to(device))
            label = torch.mm(inputs, torch.tensor([[0.0], [0.0], [1.0]], requires_grad=True).to(device))
            outputs = self.forward(input)
            loss = self.loss_func(input, outputs, label).mean()
            print("Current loss:", loss)
            self.optimiser.zero_grad()
            loss.backward(retain_graph=True)
            return loss
        self.optimiser.step(closure)
        self.optimiser = torch.optim.LBFGS(self.parameters(), lr=self.lr)

#编写目标函数解析式
templambda = []
def f(x):
    return math.tan(l*x)+(x/h)
for i in range(1, 101):
    temparray = [(i*(math.pi))/(2*l), (3*i*(math.pi))/(2*l)]
    for j in range(0, 1000):
        if(f((temparray[0]+temparray[1])/2.0)<0):
            temparray[0] = (temparray[0]+temparray[1])/2.0
        elif(f((temparray[0]+temparray[1])/2.0)>0):
            temparray[1] = (temparray[0]+temparray[1])/2.0
        else:
            templambda.append((temparray[0]+temparray[1])/2.0)
            break
    templambda.append((temparray[0]+temparray[1])/2.0)
def u(t, x):
    temp = 0.0
    for n in range(0, 100):
        def g(m):
            return (math.sin(m))*(math.sin(templambda[n]*m))
        temp += (integrate.quad(g, 0, l)[0])*(math.exp((-(a**2))*(templambda[n]**2)*t))*(math.sin(templambda[n]*x))/((l/2.0)+(h/(2*(h**2)+2*(templambda[n]**2))))
    return temp

#生成训练集
temptrainlist=[]
    #内部取样点
for i in range(0, 10000):
    t = random.uniform(5, 10)
    x = random.uniform(0, l)
    z = u(t, x)
    temptrainlist.append([t, x, z])
    #三侧边缘随机取样点
for i in range(0, 2000):
    t = random.uniform(5, 10)
    z = u(t, 0)
    temptrainlist.append([t, 0, z])
for i in range(0, 2000):
    t = random.uniform(5, 10)
    z = u(t, l)
    temptrainlist.append([t, l, z])
for i in range(0, 2000):
    x = random.uniform(0, l)
    z = u(5, x)
    temptrainlist.append([0, x, z])
trainlist = torch.tensor(temptrainlist, requires_grad=True)
train_loader = Data.DataLoader(dataset=trainlist, batch_size=BATCH_SIZE, shuffle=True)

#开始训练
PINNsModel = PINNs().to(device)
for epoch in range(1, EPOCH+1):
    for i, data in enumerate(train_loader, 0):
        torch.cuda.empty_cache()
        data=data.to(device)
        PINNsModel.train(data)
        print("Batch", i, "has been used in training")
    if((epoch % 10) == 0):
        PINNsModel.lr *= 0.9
    print("Epoch", epoch, "has finished, and the current learning rate is", PINNsModel.lr)
torch.save(PINNsModel.state_dict(), "./heat equation net.pkl")

#绘制精确解图像
x_slice = 100
y_slice = 100
t = np.linspace(5, 10, x_slice, dtype=np.float32)
x = np.linspace(0, l, y_slice, dtype=np.float32)
T, X = np.meshgrid(t, x)
Z = np.array(np.arange(0, 1, (1.0/(x_slice*y_slice)))).reshape(x_slice, y_slice)
for i in range(0, x_slice):
    for j in range(0, y_slice):
        Z[i][j] = u(T[i][j], X[i][j])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, Z, cmap=cm.YlGnBu_r)
ax.set_zlim3d(0, 2)
plt.savefig("heat equation[exact solution].pdf")

#绘制拟合函数图像并创建测试集对比训练结果与目标函数的差异
x_slice = 100
y_slice = 100
t = np.linspace(5, 10, x_slice, dtype=np.float32)
x = np.linspace(0, l, y_slice, dtype=np.float32)
T, X = np.meshgrid(t, x)
Z = np.array(np.arange(0, 1, (1.0/(x_slice*y_slice)))).reshape(x_slice, y_slice)
for i in range(0, x_slice):
    for j in range(0, y_slice):
        temp = torch.tensor([T[i][j], X[i][j]], requires_grad=True).to(device)
        Z[i][j] = PINNsModel.forward(temp).tolist()[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, Z, cmap=cm.YlGnBu_r)
ax.set_zlim3d(0, 2)
plt.savefig("heat equation[fitting solution].pdf")
error = []
for i in range(1, 10001):
    t = random.uniform(5, 10)
    x = random.uniform(0, l)
    temp = torch.tensor([t, x], requires_grad=True).to(device)
    error.append((PINNsModel.forward(temp).tolist()[0]-u(t, x))**2)
print("Mean squared error of the model:", np.mean(error))
print("Pending parameter:", (PINNsModel.mylambda[0]).tolist()[0])
print("Error of the pending parameter:", a-((PINNsModel.mylambda[0]).tolist()[0]))