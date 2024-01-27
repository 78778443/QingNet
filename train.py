import datetime
import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import YellowDataset
from net import NetV2
from tools import tools

DEVICE = 'cuda:0'
# DEVICE = 'cpu'


class Train:
    def __init__(self, root, weight_path, train=True, test=False):
        self.train = train
        self.test = test
        # 初始化函数
        self.summaryWriter = SummaryWriter('logs')  # 创建SummaryWriter对象
        self.train_dataset = YellowDataset(root=root)  # 创建训练数据集
        self.test_dataset = YellowDataset('data/test')  # 创建测试数据集
        # 创建训练数据加载器
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
        # 创建测试数据加载器
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=50, shuffle=True)
        # 创建神经网络对象
        self.net = NetV2().to(DEVICE)
        # 如果权重路径存在
        if os.path.exists(weight_path):
            # 加载模型权重
            self.net.load_state_dict(torch.load(weight_path))
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())
        # 创建用于标签的损失函数
        self.label_loss = nn.BCEWithLogitsLoss()

        # 创建用于位置的损失函数
        self.position_loss = nn.MSELoss()
        # 创建用于排序的损失函数
        self.sort_loss = nn.CrossEntropyLoss()

    def __call__(self):  # 调用函数

        if self.train:  # 若训练数据为真
            self.train_func()
        # 若测试数据为真
        if self.test:
            self.train_test()

    # 对测试数据加载器枚举
    def train_func(self):
        index1, index2 = 0, 0  # 初始化index1和index2为0
        for epoch in range(1000):  # 1000代训练
            for i, (img, label, position, sort, img_path) in enumerate(self.train_dataloader):  # 对训练数据加载器进行枚举
                # 转换数据为指定设备（DEVICE）
                img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE)
                # 获取模型输出的标签、位置和排序
                out_label, out_position, out_sort, confidence = self.net(img)
                position_loss = self.position_loss(out_position, position)  # 计算位置损失
                out_sort = out_sort[torch.where(sort >= 0)]  # 获取有效数据的排序输出
                sort = sort[torch.where(sort >= 0)]  # 获取有效数据的排序
                sort_loss = self.sort_loss(out_sort, sort)  # 计算排序损失
                label_loss = self.label_loss(out_label, label)  # 计算标签损失

                train_loss = 0.2 * label_loss + position_loss * 0.6 + 0.2 * sort_loss  # 计算训练损失
                self.opt.zero_grad()  # 梯度清零
                train_loss.backward()  # 反向传播计算梯度
                self.opt.step()  # 梯度更新
                if i % 10 == 0:  # 每100轮迭代输出
                    print(f'train_loss {i}===>>', train_loss.item())  # 输出训练损失
                    self.summaryWriter.add_scalar('train_loss', train_loss, index1)  # 写入训练损失
                    index1 += 1  # index1累加
            # 保存模型,模型名字获取日期时间信息
            date_time = str(datetime.datetime.now()).replace(' ', '-').replace(':', '_').replace('.', '_')
            torch.save(self.net.state_dict(), f'param/{date_time}-{epoch}.pt')
            sum_acc = 0  # 准确率求和初始化

            # 对测试数据加载器枚举
            for i, (img, label, position, sort, img_path) in enumerate(self.test_dataloader):
                # 转换数据为指定设备（DEVICE）
                img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE)
                # 获取模型输出的标签、位置和排序
                out_label, out_position, out_sort, confidence = self.net(img)
                # 获取有效数据的排序输出
                out_sort = out_sort[torch.where(sort >= 0)]
                # 获取有效数据的排序
                sort = sort[torch.where(sort >= 0)]
                # 计算位置损失
                position_loss = self.position_loss(out_position, position)
                # 计算排序损失
                sort_loss = self.sort_loss(out_sort, sort)
                # 计算标签损失
                label_loss = self.label_loss(out_label, label)
                # 计算测试损失
                test_loss = 0.2 * label_loss + position_loss * 0.6 + 0.2 * sort_loss
                # 计算排序的准确率
                sort_acc = torch.mean(torch.eq(sort, torch.argmax(torch.softmax(out_sort, dim=1), dim=1)).float())
                # 增加排序准确率到总准确率
                sum_acc += sort_acc
                # 每5轮迭代输出
                if i % 5 == 0:
                    # 输出测试损失
                    print(f'test_loss {i}===>>', test_loss.item())
                    # 写入测试损失
                    self.summaryWriter.add_scalar('test_loss', test_loss, index2)
                    # indxe2累加
                    index2 += 1
            avg_acc = sum_acc / i  # 计算平均准确率
            print(f'sort_acc {i}==>>', avg_acc)  # 输出平均准确率
            self.summaryWriter.add_scalar('avg_acc', avg_acc, epoch)  # 写入平均准确率

    def train_test(self):
        # 准确率求和初始化
        sum_acc = 0
        # 对测试数据加载器枚举
        for i, (img, label, position, sort, img_path) in enumerate(self.test_dataloader):
            # 转换数据为指定设备（DEVICE）
            img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE)
            # 获取模型输出的标签、位置和排序
            out_label, out_position, out_sort, confidence = self.net(img)
            # 标签输出为sigmoid函数输出
            out_label = torch.sigmoid(out_label)
            # 位置输出归一化
            out_position = out_position * 300
            # 位置数据变换为整形
            out_position = [int(i) for i in out_position[0]]
            # 排序输出为one-hot编码
            out_sort = torch.argmax(torch.softmax(out_sort, dim=1))

            # 位置标签归一化
            position = position * 300
            position = [int(i) for i in position[0]]  # 位置数据变换为整形
            # 直接使用获取到的图像文件路径
            img_path = img_path[0]  # 注意：img_path是一个包含单个元素的列表

            # 调用view_image
            tools.view_image(img_path, label, position, sort, out_label, out_position, out_sort)


if __name__ == '__main__':
    param = ''
    if os.listdir('param/'):
        param = 'param/' + max(os.listdir('param/'))
    train = Train('data/train', param,True,False)
    # train = Train('data/train', param,False,True)
    train()
