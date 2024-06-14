#定义两个用于深度学习的自定义损失函数类，BCEDiceLoss和LovaszHingeLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 尝试导入lovasz_hinge损失函数，如果失败则忽略
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

class BCEDiceLoss(nn.Module):
    def __init__(self):
        # 初始化函数，无需额外操作
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        # 前向传播函数，计算损失值
        bce = F.binary_cross_entropy_with_logits(input, target)  # 二元交叉熵损失
        smooth = 1e-5  # 为了避免除零错误加入的小常数
        input = torch.sigmoid(input)  # 将输入通过sigmoid函数转换
        num = target.size(0)  # 获取批次大小
        # 重塑input和target的形状以匹配
        input = input.view(num, -1)
        target = target.view(num, -1)
        # 计算交并比的分子部分
        intersection = (input * target)
        # 计算dice系数
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        # 计算dice损失
        dice = 1 - dice.sum() / num
        # 返回结合了bce和dice的总损失
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        # 初始化函数，无需额外操作
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        # 前向传播函数，计算损失值
        input = input.squeeze(1)  # 压缩输入的维度，移除单维度特征通道
        target = target.squeeze(1)  # 压缩目标的维度
        # 计算lovasz hinge损失
        loss = lovasz_hinge(input, target, per_image=True)
        return loss
