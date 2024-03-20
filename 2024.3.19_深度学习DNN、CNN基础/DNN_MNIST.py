import torch           #导入pyTorch库
import torch.nn as nn  #torch.nn提供搭建网络所需要的组件
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 展示高清图
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

# 制作数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

# 下载训练集与测试集
train_Data = datasets.MNIST(
    root = 'D:\Pycharm\Py_Projects\DNN.CNN',
    train = True,
    download = True,
    transform = transform
)
test_Data = datasets.MNIST(
    root = 'D:\Pycharm\Py_Projects\DNN.CNN',
    train = False,
    download = True,
    transform = transform
)

# 批次加载器
train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
test_loader  = DataLoader(test_Data, shuffle=False, batch_size=64)

# 第二步.构建神经网络
class DNN(nn.Module):

    def __init__(self):
        ''' 搭建神经网络各层 '''
        super(DNN,self).__init__()
        self.net = nn.Sequential(            # 按顺序搭建各层
            nn.Flatten(),                    # 先把图像铺平成一维
            nn.Linear(784, 512), nn.ReLU(),  # 第1层：全连接层，后接激活函数
            nn.Linear(512, 256), nn.ReLU(),  # 第2层：全连接层
            nn.Linear(256, 128), nn.ReLU(),  # 第3层：全连接层
            nn.Linear(128, 64),  nn.ReLU(),  # 第4层：全连接层
            nn.Linear(64, 10),               # 第5层：全连接层
        )

    def forward(self, x):
        ''' 前向传播 '''
        y = self.net(x)        # x即输入数据
        return y               # y即输出数据

model = DNN()#.to('cuda:0')    # 创建子类的实例，并搬到GPU上

# 损失函数的选择
loss_fn = nn.CrossEntropyLoss()    # 自带softmax激活函数
# 优化算法的选择
learning_rate = 0.01    # 设置学习率
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
    momentum = 0.5
)

# 训练网络
epochs = 5
losses = []        # 记录损失函数变化的列表

for epoch in range(epochs):
    for (x, y) in train_loader:                  # 获取小批次的x与y
#         x, y = x.to('cuda:0'), y.to('cuda:0')    # 把小批次搬到GPU上
        Pred = model(x)                          # 一次前向传播（小批量）
        loss = loss_fn(Pred, y)                  # 计算损失函数
        losses.append(loss.item())               # 记录损失函数的变化
        optimizer.zero_grad()                    # 清理上一轮滞留的梯度
        loss.backward()                          # 一次反向传播
        optimizer.step()                         # 优化内部参数

Fig = plt.figure()
plt.plot(range(len(losses)), losses)
plt.ylabel('loss')
plt.xlabel('len(losses)')
plt.show()

# 测试网络
correct = 0
total = 0

with torch.no_grad():  # 该局部关闭梯度计算功能
    for (x, y) in test_loader:  # 获取小批次的x与y
        #         x, y = x.to('cuda:0'), y.to('cuda:0')           # 把小批次搬到GPU上
        Pred = model(x)  # 一次前向传播（小批量）
        _, predicted = torch.max(Pred.data, dim=1)
        correct += torch.sum((predicted == y))
        total += y.size(0)

print(f'测试集精准度: {100 * correct / total} %')
