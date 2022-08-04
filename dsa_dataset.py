import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv

random.seed(100)


class DSADataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.data_info = list()
        img_dirs = os.listdir(img_dir)
        label_dirs = os.listdir(label_dir)
        for i in range(len(img_dirs)):
            ipath = os.path.join(img_dir, str(i) + ".png")
            lpath = os.path.join(label_dir, str(i) + ".png")
            self.data_info.append((ipath, lpath))

    def __getitem__(self, item):
        img_path, label_path = self.data_info[item]
        img = Image.open(img_path)
        label = Image.open(label_path)
        # array_img = np.asarray(img)  # 转换为数组形式
        # data_img = torch.from_numpy(array_img)  # array转换为tensor
        # array_label = np.asarray(label)  # 转换为数组形式
        # data_label = torch.from_numpy(array_label)  # array转换为tensor
        transform = transforms.ToTensor()
        img = transform(img)
        label = transform(label)
        # data_img = torch.unsqueeze(data_img, 0)
        # data_label = torch.unsqueeze(data_label, 0)
        # return DSADataset.adjustData(img, label)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def adjustData(img, mask):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return img, mask


def main():
    img_dir = 'D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/image'
    label_dir = 'D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/label'
    d = DSADataset(img_dir, label_dir)
    dataloader = DataLoader(d, 1, True, num_workers=0)
    dataitem = iter(d)
    imgs, labels = next(dataitem)
    print(labels)
    print(labels.shape)
    print(imgs)
    print(imgs.shape)


# def main():
#     torch.set_printoptions(threshold=np.inf) # 将tensor显示完全
#     label_dir = 'D:/JupyterNotebook/Unet_GAN_demo/data/membrane/train/label/0.png'
#     label = Image.open(label_dir)
#     # label.show()
#     label = label.convert('L')
#     array_label = np.asarray(label)  # 转换为数组形式
#     data_label = torch.from_numpy(array_label)  # array转换为tensor
#     print(data_label)

if __name__ == '__main__':
    main()
