import os

import torch

from net import NetV2
import cv2

# 判断是否在主程序中运行
if __name__ == '__main__':
    # 实例化一个NetV2模型
    model = NetV2()
    # 读取预训练参数
    model.load_state_dict(torch.load('param/2021-07-23-20_27_07_213811-11.pt'))
    # 测试数据集的根目录
    root = 'data/test/'
    # 遍历测试数据集中的文件
    for i in os.listdir(root):
        # 读取图像数据
        img = cv2.imread(root + i)
        # 将图像数据转换成PyTorch张量，并进行通道维度的变换
        img_data = torch.tensor(img).permute(2, 0, 1)
        # 在第0维度上增加一个维度，并归一化
        img_data = torch.unsqueeze(img_data, dim=0) / 255
        # 将图像数据输入模型进行预测
        rst = model(img_data)
        # 对预测结果进行sigmoid处理
        label = torch.sigmoid(rst[0])
        # 对预测结果进行softmax处理
        sort = torch.softmax(rst[2], dim=1)
