


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


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma, gpu_id):
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
    soft_hist = SoftHistogram(bins=bit, min=0, max=1, sigma=400, gpu_id=gpu_id)
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

def portrait_l1_loss(sr, gt, seg_map, person_id=36, eps=1e-6):
    """
    Count Portrait-region L1 loss
    """
    if seg_map is None:
        # 没有分割图就直接不给梯度
        return sr.new_tensor(0.0)

    # 1) 根据 seg_map 得到每个像素的类别
    if seg_map.dim() == 4:
        # B x C_seg x H' x W' → argmax 得到类别 id
        seg_cls = seg_map.argmax(dim=1, keepdim=True).float()  # B x 1 x H' x W'
    elif seg_map.dim() == 3:
        # B x H' x W'
        seg_cls = seg_map.unsqueeze(1).float()                  # B x 1 x H' x W'
    else:
        raise ValueError(f"Unexpected seg_map shape: {seg_map.shape}")

    # 2) 取出 person 类 (id=36) 的 mask
    mask = (seg_cls == float(person_id)).float()                # B x 1 x H' x W'

    # 3) 尺寸对齐到 sr / gt
    if mask.shape[-2:] != sr.shape[-2:]:
        mask = F.interpolate(mask, size=sr.shape[-2:], mode='nearest')

    # 4) 扩展到通道维度
    mask = mask.expand_as(sr)                                  # B x C x H x W

    valid = mask.sum()
    if valid.item() < 1:
        # 这一 batch 里没有人像像素，损失记 0
        return sr.new_tensor(0.0)

    # 5) 只在 mask 内算 L1
    diff = (sr - gt).abs() * mask
    loss = diff.sum() / (valid + eps)
    return loss