import os
import torch
import cv2
from torch.utils.data import DataLoader

from data import YellowDataset
from net import NetV2

from tools import tools


class Predictor:
    def __init__(self, weight_path):
        self.net = NetV2()
        self.net.load_state_dict(torch.load(weight_path))
        self.net.eval()

    def predict(self, img_path):
        img = cv2.imread(img_path)
        img_data = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out_label, out_position, out_sort, confidence = self.net(img_data)
        label = torch.sigmoid(out_label).item()
        position = (out_position * 300).squeeze().int().tolist()
        sort = torch.argmax(torch.softmax(out_sort, dim=1)).item()
        return label, position, sort,confidence


if __name__ == '__main__':
    predictor = Predictor('param/' + max(os.listdir('param/')))
    test_dataset = YellowDataset('data/test')  # 创建测试数据集
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for i, (img, label, position, sort, img_path) in enumerate(test_dataloader):
        # 直接使用获取到的图像文件路径
        img_path = img_path[0]  # 注意：img_path是一个包含单个元素的列表
        out_label, out_position, out_sort,confidence = predictor.predict(img_path)

        # 位置标签归一化
        position = position * 300
        position = [int(i) for i in position[0]]  # 位置数据变换为整形

        print(position, out_position,confidence)
        # 调用view_image
        tools.view_image(img_path, label, position, sort, out_label, out_position, out_sort)
