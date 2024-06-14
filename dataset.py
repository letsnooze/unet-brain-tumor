#数据加载

import numpy as np
import cv2  # OpenCV库，用于图像处理
import random
from skimage.io import imread  # skimage库中的函数，用于读取图像
from skimage import color  # skimage库中的模块，用于图像颜色空间转换
import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):  # 继承自torch的Dataset类

    def __init__(self, args, img_paths, mask_paths, aug=False):
        # 初始化函数，存储图像和掩码的路径，以及是否进行数据增强的选项
        self.args = args
        self.img_paths = img_paths  # 存储图像文件路径的列表
        self.mask_paths = mask_paths  # 存储掩码文件路径的列表
        self.aug = aug  # 数据增强的开关

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 根据索引idx获取一个样本
        img_path = self.img_paths[idx]  # 获取图像路径
        mask_path = self.mask_paths[idx]  # 获取掩码路径

        # 读取numpy数组数据
        npimage = np.load(img_path)  # 加载图像数据
        npmask = np.load(mask_path)  # 加载掩码数据
        npimage = npimage.transpose((2, 0, 1))  # 调整维度顺序

        # 根据掩码生成不同的标签
        WT_Label = (npmask == 1) | (npmask == 2) | (npmask == 4)  # 白质标签
        TC_Label = (npmask == 1) | (npmask == 4)  # 肿瘤核心标签
        ET_Label = npmask == 4  # 肿瘤边缘标签

        # 组合标签
        nplabel = np.stack((WT_Label, TC_Label, ET_Label), axis=-1).astype("float32")
        nplabel = nplabel.transpose((2, 0, 1))  # 调整维度顺序

        # 类型转换
        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage, nplabel

        # 注释掉的代码是读取普通图片文件的示例，当前实现使用的是npy文件

        #读图片（如jpg、png）的代码
        '''
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = color.gray2rgb(image)
        #image = image[:,:,np.newaxis]
        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))       
        return image, mask
        '''
