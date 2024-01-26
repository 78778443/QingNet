import os

import numpy as np
import torch

from net import NetV2
import cv2


def view_image(self, img, position, sort, out_label, out_position, out_sort):
    # 压缩图像
    new_img = torch.squeeze(img)
    # 转换图像排列方式
    new_img = new_img.permute(1, 2, 0)
    # 转换为numpy数组
    new_img = np.array(new_img.cpu())
    new_img = new_img.copy()  # 图像复制
    # 在图像上绘制矩形框
    cv2.rectangle(new_img, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 3)
    # 在图像上绘制文本信息
    cv2.putText(new_img, str(sort.item()), (position[0], position[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 3)
    # 如果标签输出大于0.5
    if out_label.item() > 0.5:
        # 绘制预测框
        cv2.rectangle(new_img, (out_position[0], out_position[1]), (out_position[2], out_position[3]),
                      (0, 0, 255), 3)
        # 绘制预测文本
        cv2.putText(new_img, str(out_sort.item()), (out_position[0], out_position[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('new_img', new_img)  # 显示图像
    cv2.waitKey(500)  # 延迟显示
    cv2.destroyAllWindows()  # 关闭所有图板


# 判断是否在主程序中运行
if __name__ == '__main__':
    # 实例化一个NetV2模型
    model = NetV2()
    # 读取预训练参数
    model.load_state_dict(torch.load('param/2023-12-23-15_16_09_131911-35.pt'))
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
