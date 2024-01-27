from torch import nn
import torch


class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()  # 初始化Net_v2类，继承自父类的初始化方法
        self.layers = nn.Sequential(  # 创建网络的序列结构
            nn.Conv2d(3, 16, 3),  # 2维卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(3),  # 最大池化层，池化核大小为3x3
            nn.Conv2d(16, 22, 3),  # 2维卷积层，输入通道数为16，输出通道数为22，卷积核大小为3x3
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2),  # 最大池化层，池化核大小为2x2
            nn.Conv2d(22, 32, 5),  # 2维卷积层，输入通道数为22，输出通道数为32，卷积核大小为5x5
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2),  # 最大池化层，池化核大小为2x2
            nn.Conv2d(32, 64, 5),  # 2维卷积层，输入通道数为32，输出通道数为64，卷积核大小为5x5
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2),  # 最大池化层，池化核大小为2x2
            nn.Conv2d(64, 82, 3),  # 2维卷积层，输入通道数为64，输出通道数为82，卷积核大小为3x3
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(82, 128, 3),  # 2维卷积层，输入通道数为82，输出通道数为128，卷积核大小为3x3
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(128, 25, 3),  # 2维卷积层，输入通道数为128，输出通道数为25，卷积核大小为3x3
            nn.ReLU()  # ReLU激活函数
        )

        self.confidence_layer = nn.Sequential(
            nn.Conv2d(25, 1, 3),  # 为置信度分数添加一个通道
            nn.Sigmoid(),  # 应用 Sigmoid 激活以确保值在 0 到 1 之间
        )

        self.label_layer = nn.Sequential(  # 标签层
            nn.Conv2d(25, 1, 3),  # 2维卷积层，输入通道数为25，输出通道数为1，卷积核大小为3x3
            nn.ReLU(),  # ReLU激活函数

        )
        self.position_layers = nn.Sequential(  # 位置层
            nn.Conv2d(25, 4, 3),  # 2维卷积层，输入通道数为25，输出通道数为4，卷积核大小为3x3
            nn.ReLU()  # ReLU激活函数
        )
        self.sort_layers = nn.Sequential(  # 排序层
            nn.Conv2d(25, 20, 3),  # 2维卷积层，输入通道数为25，输出通道数为20，卷积核大小为3x3
            nn.ReLU()  # ReLU激活函数
        )

    def forward(self, x):
        # 通过网络层计算输出
        out = self.layers(x)
        # 使用label层获取标签
        label = self.label_layer(out)
        # 去除多余的维度
        label = torch.squeeze(label, dim=2)
        label = torch.squeeze(label, dim=2)
        label = torch.squeeze(label, dim=1)

        confidence = self.confidence_layer(out)  # 新增：置信度分数
        confidence = torch.squeeze(confidence, dim=2)
        confidence = torch.squeeze(confidence, dim=2)

        # 使用position层获取位置信息
        position = self.position_layers(out)
        position = torch.squeeze(position, dim=2)
        position = torch.squeeze(position, dim=2)
        # 使用sort层获取排序信息
        sort = self.sort_layers(out)
        sort = torch.squeeze(sort, dim=2)
        sort = torch.squeeze(sort, dim=2)
        # 返回label, position, sort
        return label, position, sort, confidence


if __name__ == '__main__':
    net = NetV2()
    x = torch.randn(3, 3, 300, 300)
    print(net(x)[2].shape)
