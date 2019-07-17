import torch
import pandas as pd
import numpy as np
import math
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image, ImageEnhance

img_w = 1024
img_h = 1024


def enhance(xb):
    # random do the input image augmentation
    if np.random.random() > 0.5:
        # sharpness
        factor = 0.5 + np.random.random()
        xb = ImageEnhance.Sharpness(xb).enhance(factor)
    if np.random.random() > 0.5:
        # color augument
        factor = 0.5 + np.random.random()
        xb = ImageEnhance.Color(xb).enhance(factor)
    if np.random.random() > 0.5:
        # contrast augument
        factor = 0.5 + np.random.random()
        xb = ImageEnhance.Contrast(xb).enhance(factor)
    if np.random.random() > 0.5:
        # brightness
        factor = 0.5 + np.random.random()
        xb = ImageEnhance.Brightness(xb).enhance(factor)
    return xb


def rotate(xb, yb):
    transtypes = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                  Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
    transtype = transtypes[np.random.randint(len(transtypes))]
    xb = xb.transpose(transtype)  # flipcode > 0：沿y轴翻转
    yb = yb.transpose(transtype)
    return xb, yb


# def blur(img):
#    img = cv2.blur(img, (3, 3));
#    return img


# def add_noise(img):
#    for i in range(200):  # 添加点噪声
#        temp_x = np.random.randint(0, img.shape[0])
#        temp_y = np.random.randint(0, img.shape[1])
#        img[temp_x][temp_y] = 255
#    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb)

    if np.random.random() < 0.25:
        xb = enhance(xb)

    return xb, yb


class AgricultureDataset(Dataset):
    def __init__(self, image_path: str, label_path: str, datalist, class_name:int, mode="train", train_ratio=0.9,
                 is_aug=True):

        self.image_path = image_path + "/"
        self.label_path = label_path + "/"
        self.data_list = datalist
        self.mode = mode
        self.is_aug = is_aug
        self.class_name = class_name
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
        label = np.transpose(label, (2, 0, 1))
        label = label[self.class_name, :, :]
        label = torch.from_numpy(label).float()

        return img, label

if __name__=="__main__":
    image_path=""
    image=Image.read(image_path)
    image_eh=ehance(image)


