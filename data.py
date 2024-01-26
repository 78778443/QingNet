import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
import os
from torch.nn.functional import one_hot
from PIL import Image


class YellowDataset(Dataset):
    # 初始化方法，传入root参数
    def __init__(self, root):
        # 创建一个空的数据集列表
        self.dataset = []
        # 遍历root目录下的所有文件名
        for filename in os.listdir(root):
            # 将文件名与root路径拼接起来，并添加到数据集列表中
            self.dataset.append(os.path.join(root, filename))

    def __len__(self):
        # 返回数据集的长度
        return len(self.dataset)

    # 获取数据集中特定索引的数据项
    def __getitem__(self, index):
        # 获取索引对应的数据
        data = self.dataset[index]
        # 读取图像数据
        img = cv2.imread(data)
        # 对图像数据进行归一化处理
        img = img / 255
        # 转换为PyTorch张量并对维度进行排列
        img = torch.tensor(img).permute(2, 0, 1)
        # 将数据按'.'分割为列表
        data_list = data.split('.')
        # 获取标签
        label = int(data_list[1])
        # 获取位置信息
        position = data_list[2:6]
        # 对位置信息进行归一化处理
        position = [int(i) / 300 for i in position]
        # 获取排序信息
        sort = int(data_list[6]) - 1

        # 返回处理后的图像数据、标签、位置信息和排序信息
        return np.float32(img), np.float32(label), np.float32(position), sort


if __name__ == '__main__':
    data = YellowDataset('data/train')
    print(data[1][3])
