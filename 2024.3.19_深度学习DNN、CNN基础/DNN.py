import torch            #导入pyTorch库
import torch.nn as nn   #torch.nn提供搭建网络所需要的组件
import matplotlib.pyplot as plt

# 展示高清图
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

# 第一步.生成数据集
# torch.rand()生成区间[0,1）内均匀分布的一组随机数Tensor
X1 = torch.rand(10000,1)    # 输入特征1
X2 = torch.rand(10000,1)    # 输入特征2
X3 = torch.rand(10000,1)    # 输入特征3
Y1 = ( (X1+X2+X3)<1 ).float()     #.float()将布尔型张量转化为浮点型张量。 # 输出特征1
Y2 = ( (1<(X1+X2+X3)) & ((X1+X2+X3)<2) ).float()     # 输出特征2
Y3 = ( (X1+X2+X3)>2 ).float()                        # 输出特征3
Data = torch.cat([X1,X2,X3,Y1,Y2,Y3],axis=1)         #数据拼接

# 划分训练集与测试集
train_size = int(len(Data) * 0.8)      # 训练集的样本数量
test_size  = len(Data) - train_size    # 测试集的样本数量
Data = Data[torch.randperm( Data.size(0)) , : ]    # 打乱样本的顺序
train_Data = Data[:train_size, :]    # 训练集样本
test_Data  = Data[train_size:, :]    # 测试集样本

# 第二步.构建神经网络
class DNN(nn.Module):

    def __init__(self):
        ''' 搭建神经网络各层 '''
        super(DNN,self).__init__()
        self.net = nn.Sequential(          # 按顺序搭建各层
            nn.Linear(3, 5), nn.ReLU(),    # 第1层：全连接层，后接激活函数
            nn.Linear(5, 5), nn.ReLU(),    # 第2层：全连接层
            nn.Linear(5, 5), nn.ReLU(),    # 第3层：全连接层
            nn.Linear(5, 3)                # 第4层：全连接层
        )

    def forward(self, x):
        ''' 前向传播 '''
        y = self.net(x)    # x即输入数据
        return y           # y即输出数据

model = DNN()#.to('cuda:0')     # 创建子类的实例，并搬到GPU上

# 损失函数的选择
loss_fn = nn.MSELoss()
# 优化算法的选择
learning_rate = 0.01    # 设置学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 第三步.训练网络
epochs = 1000
losses = []      # 记录损失函数变化的列表

# 给训练集划分输入与输出
X = train_Data[:, :3]      # 前3列为输入特征
Y = train_Data[:, -3:]     # 后3列为输出特征

for epoch in range(epochs):       #range(epochs)是[0,1,2,...,epochs]的列表
    Pred = model(X)               # 一次前向传播（BGD），预测值Pred
    loss = loss_fn(Pred, Y)       # 计算损失函数
    losses.append(loss.item())    # 记录损失函数的变化
    optimizer.zero_grad()         # 清理上一轮滞留的梯度
    loss.backward()               # 一次反向传播
    optimizer.step()              # 优化内部参数

Fig = plt.figure()                #做Loss和epoch的图
plt.plot(range(epochs), losses)   #plt.plot（纵轴量，横轴量）
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# 第四步.测试网络
# 给测试集划分输入与输出
X = test_Data[:, :3]      # 前3列为输入特征
Y = test_Data[:, -3:]     # 后3列为输出特征

with torch.no_grad():     # 该局部关闭梯度计算功能
    Pred = model(X)       # 一次前向传播（批量）
    Pred[:,torch.argmax(Pred, axis=1)] = 1
    Pred[Pred!=1] = 0
    correct = torch.sum( (Pred == Y).all(1) )    # 预测正确的样本
    total = Y.size(0)                            # 全部的样本数量
    print(f'测试集精准度: {100*correct/total} %')