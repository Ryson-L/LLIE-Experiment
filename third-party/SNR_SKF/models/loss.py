import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers.cuda()

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        # y = x.sum()
        # x = x / (x.sum() + 0.0001)
        return x

    def forward_1(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=-1)
        # x = x / (x.sum() + 0.00001)
        return x

def hist_loss(seg_pred, input_1, input_2, gpu_id=None):
    '''
    1. seg_pred transform to [1,2,3,2,3,1,3...] x batchsize
    2. Get class 1,2,3 index
    3. Use index to get value of img1 and img2
    4. Get hist of img1 and img2
    :return:
    '''
    Charloss = CharbonnierLoss()
    N, C, H, W = seg_pred.shape
    bit = 256
    seg_pred = seg_pred.reshape(N, C, -1)
    seg_pred_cls = seg_pred.argmax(dim=1)
    input_1 = input_1.reshape(N, 3, -1)
    input_2 = input_2.reshape(N, 3, -1)
    soft_hist = SoftHistogram(bins=bit, min=0, max=1, sigma=400)
    loss = []
    # img:4,3,96,96  hist:4,9,256
    for n in range(N):
        # TODO 简化
        cls = seg_pred_cls[n]  # (H * W)
        img1 = input_1[n]
        img2 = input_2[n]
        # loss1 = soft_hist(img1[0])
        for c in range(C):
            cls_index = torch.nonzero(cls == c).squeeze()
            img1_index = img1[:, cls_index]
            img2_index = img2[:, cls_index]
            for i in range(img1.shape[0]):
                img1_hist = soft_hist(img1_index[i])
                img2_hist = soft_hist(img2_index[i])
                loss.append(F.l1_loss(img1_hist, img2_hist))
    loss = sum(loss) / (N*H*W)
    return loss



def portrait_l1_loss(seg_pred, input_1, input_2, person_id, eps=1e-6):
    """
    只在人像区域内计算平均 L1 误差。
    seg_pred: [N, C, H, W]  语义分割 logits 或概率图
    input_1: [N, 3, H, W]  通常是 fake / enhanced
    input_2: [N, 3, H, W]  通常是 GT
    person_id: int         HRNet 中 'person' 对应的类别 ID
    """
    # seg_pred -> 每个像素的类别标签 [N, H, W]
    N, C, H, W = seg_pred.shape
    seg_flat = seg_pred.reshape(N, C, -1)
    seg_cls = seg_flat.argmax(dim=1)          # [N, H*W]
    seg_cls = seg_cls.view(N, 1, H, W)        # [N, 1, H, W]

    # 生成 person 区域 mask
    mask = (seg_cls == person_id).float()     # [N, 1, H, W]
    # 扩展到 3 通道
    if input_1.size(1) == 3 and mask.size(1) == 1:
        mask = mask.repeat(1, 3, 1, 1)

    # 对齐尺寸（以防万一）
    if mask.shape[2:] != input_1.shape[2:]:
        mask = F.interpolate(mask, size=input_1.shape[2:], mode='nearest')

    # 只在人像区域内算 L1
    diff = torch.abs(input_1 - input_2) * mask
    denom = mask.sum() + eps
    if denom.item() == 0:
        # 当前 batch 没有人像像素，就返回 0，让外面自己判断要不要加权
        return torch.tensor(0.0, device=input_1.device)

    loss = diff.sum() / denom
    return loss
