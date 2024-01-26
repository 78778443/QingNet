import os
import torch
import cv2
import numpy as np
from net import NetV2
from PIL import Image

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
            out_label, out_position, out_sort = self.net(img_data)
        label = torch.sigmoid(out_label).item()
        position = (out_position * 300).squeeze().int().tolist()
        sort = torch.argmax(torch.softmax(out_sort, dim=1)).item()
        return label, position, sort


if __name__ == '__main__':
    predictor = Predictor('param/2023-12-23-15_16_09_131911-35.pt')
    test_dir = 'data/test/'

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        label, position, sort = predictor.predict(img_path)
        tools.view_image(img_path, label, position, sort, 0, [0, 0, 0, 0], 0)
