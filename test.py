import os
import time
import torch
import torch.utils
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt

from GeneralUtils import TrainUtils
from Constants import Metrics




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 定义一个Sequential容器，按顺序执行其中的所有模块
        self.model = nn.Sequential(
            # 第一层卷积层
            # 输入通道数为1（灰度图像），输出通道数为16，卷积核大小为3x3，步长为1，填充1个像素
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # ReLU激活函数，引入非线性
            nn.ReLU(),
            # 最大池化层，池化窗口大小为2x2，步长为2
            # 通过池化操作，特征图尺寸减半，从28x28变为14x14
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积层
            # 输入通道数为16（上一层的输出通道数），输出通道数为32，卷积核大小为3x3，步长为1，填充1个像素
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # ReLU激活函数
            nn.ReLU(),
            # 最大池化层
            # 特征图尺寸再次减半，从14x14变为7x7
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层卷积层
            # 输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1，填充1个像素
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # ReLU激活函数
            nn.ReLU(),

            # 展平层，将多维张量展平成一维张量，以便输入到全连接层
            # 这里输入的特征图尺寸为7x7x64，展平后为7*7*64=3136
            nn.Flatten(),

            # 第一个全连接层
            # 输入维度为3136，输出维度为128
            nn.Linear(in_features=7 * 7 * 64, out_features=128),
            # ReLU激活函数
            nn.ReLU(),

            # 第二个全连接层
            # 输入维度为128，输出维度为10（对应MNIST数据集的10个类别）
            nn.Linear(in_features=128, out_features=10),
            # Softmax层，将输出转换为概率分布
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        # 前向传播过程
        # 将输入数据传递给定义好的模型
        output = self.model(input)
        return output



def main():
    """主函数"""
    # 如果网络能够使用GPU则使用GPU进行训练
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("使用设备：", device)

    net = Net()

    """数据准备"""
    # 这个函数包含了两个操作: 将图片转换为张量，以及将图片进行归一化处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = [0.5], std = [0.5])
    ]
    )
    # 数据保存路径
    data_path = os.path.join("data")

    # 下载训练集和测试集
    train_data = torchvision.datasets.MNIST(data_path, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(data_path, train = False, transform=transform)

    # 设置每一个Batch的大小
    batch_size = 256

    # 构建数据集和测试集的DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_data,
        batch_size = batch_size
        )
    
    # 初始化训练工具
    train_utils = TrainUtils()

    # 设置训练轮数
    EPOCHS = 3
    metrics = [Metrics.ACCURACY, Metrics.PRECISION, Metrics.RECALL, Metrics.F1_SCORE]
    # 损失函数
    loss_f = torch.nn.CrossEntropyLoss()
    # 迭代器
    optimizer = torch.optim.Adam(net.parameters())
    model, history = train_utils.general_train(
        test_data_loader=test_data_loader,
        train_data_loader=train_data_loader,
        net=net,
        metrics = metrics,
        optimizer=optimizer,
        criterion=loss_f,
        epochs=EPOCHS,
        device=device
    )
    print(history)
    train_utils.plot_history(history=history, save_path="./history.png")

main()