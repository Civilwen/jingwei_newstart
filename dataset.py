import torch
import pandas as pd
import numpy as np
import math
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


img_w = 512
img_h = 512


def gamma_transform(img, gamma):  # gamma变换
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)  # 一个左闭右开的均匀分布
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    # if np.random.random() < 0.25:
    #     xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


class AgricultureDataset(Dataset):
    def __init__(self, image_path: str, label_path: str, datalist, mode="train", train_ratio=0.7,
                 is_aug=True):
        self.image_path = image_path + "/"
        self.label_path = label_path + "/"
        self.data_list = datalist
        self.mode = mode
        self.is_aug = is_aug

        data_list = pd.read_csv(self.data_csv_path)
        data_list = data_list.sample(frac=1)

        self.data_sclae = train_ratio
        imgs = []
        labels = []
        row_numbers = self.data_list.shape[0]
        for index, row in self.data_list.iterrows():
            imgs.append(row[0])
            labels.append(row[1])

        if mode == "train":
            self.imgs = imgs[0:math.ceil(row_numbers * self.data_sclae)]
            self.labels = labels[0:math.ceil(row_numbers * self.data_sclae)]
        elif mode == "valid":
            self.imgs = imgs[math.ceil(row_numbers * self.data_sclae):]
            self.labels = labels[math.ceil(row_numbers * self.data_sclae):]
        else:
            print("Input mode should be 'train','valid'")

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.image_path + self.imgs[index]
        lab_name = self.label_path + self.labels[index]

        img = Image.open(img_name)
        label = Image.open(lab_name)

        if self.is_aug:
            img, label = data_augment(img, label)
        # ToTensor
        img = self.transforms(img)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        return img, label
