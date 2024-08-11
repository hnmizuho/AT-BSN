import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
from torch import Tensor
import torch.nn.functional as F
# from .warp import Warp, event_warp_LR, event_warp_MID, warp_dense_events, spike_warp_LR, spike_warp_MID

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3, alpha=0.5):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, warpped_img2, img1):
        delta = warpped_img2 - img1
        loss = torch.mean(torch.pow(torch.mul(delta,delta) + torch.mul(self.epsilon,self.epsilon), self.alpha))
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return nn.MSELoss()(x, target)
        elif self.losstype == 'l1':
            return nn.L1Loss()(x, target)
        else:
            raise NotImplementedError('Reconstruction loss type error!')

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics
    
class Census(nn.Module):
    '''
    adapted from smurf
    '''
    def __init__(self, use_cuda = True):
        super(Census, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()
        if use_cuda:
            self.w = self.w.cuda()

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    # def rgb2gray(self, rgb):
    #     r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
    #     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #     return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        # dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        dist_norm = torch.sum(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding): #就是把周围替换成0，size不变
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def abs_robust_loss(self, diff, eps=0.01, q=0.4):
        """The so-called robust loss used by DDFlow.""" #csy 变种char
        return torch.pow((torch.abs(diff) + eps), q)

    def forward(self, img0, img1, mask):
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        ham = self.hamming(img0, img1)
        diff = self.abs_robust_loss(ham)
        padded_mask = self.valid_mask(mask, 3)
        loss_mean = torch.sum(diff * padded_mask) / torch.sum(padded_mask.detach() + 1e-6) #1e-6 为了防止分母为0
        return loss_mean

class Second_Order_SmoothLoss(nn.Module):
    '''
    adapted from smurf
    '''
    def __init__(self):
        super(Second_Order_SmoothLoss, self).__init__()
        self.smoothness_edge_constant = 150.

    def robust_l1(self, x):
        """Robust L1 metric."""
        return (x**2 + 0.001**2)**0.5

    def image_grads(self, image_batch, stride=1):
        image_batch_gh = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
        image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
        return image_batch_gh, image_batch_gw

    def edge_weighting_fn(self, x):
        """ exponential """
        # return torch.exp(-torch.mean(abs(self.smoothness_edge_constant * x), -1, keepdims=True)) #tf 里，维度为nhwc
        return torch.exp(-torch.mean(abs(self.smoothness_edge_constant * x), 1, keepdims=True)) #通道上平均图像梯度的均值的指数

    def forward(self, image, flow):
        '''
        image: Image used for the edge-aware weighting [batch, height, width, 2].
        flow: Flow field for with to compute the smoothness loss [batch, height,
        width, 2].
        '''
        img_gx, img_gy = self.image_grads(image, stride=2)
        weights_xx = self.edge_weighting_fn(img_gx)
        weights_yy = self.edge_weighting_fn(img_gy)

        flow_gx, flow_gy = self.image_grads(flow)
        flow_gxx, _ = self.image_grads(flow_gx)
        _, flow_gyy = self.image_grads(flow_gy)

        loss = torch.mean(weights_xx * self.robust_l1(flow_gxx)) + torch.mean(weights_yy * self.robust_l1(flow_gyy))
        loss = loss / 2.

        return loss

def smurf_warp(x, flo):
    '''
    adapted from smurf
    (但实际不用乘mask, 因为默认padding_mode='zero'就相当于乘mask了..)
    输入255输出就255, 输入01输出就01 但最好01, 和mask保持一致
    '''
    B, C, H, W = x.size()
    # mesh grid, Construct a grid of the image coordinates.
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    # Add the flow field to the image grid.
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo # vgrid is so called "the warp from the flow field. The warp, i.e. the endpoints of the estimated flow. Or image coordinates." in SMURF

    valid_warp_mask = torch.logical_and(
        torch.logical_and(vgrid[:, 0, :, :] >= 0.0, vgrid[:, 0, :, :] <= H - 1),
        torch.logical_and(vgrid[:, 1, :, :] >= 0.0, vgrid[:, 1, :, :] <= W - 1)
    )
    valid_warp_mask = torch.unsqueeze(valid_warp_mask,1).type(torch.float32) # The mask showing which coordinates are valid. (SMURF)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    warpped = F.grid_sample(x, vgrid, align_corners=True)

    return warpped, valid_warp_mask

    # mask = torch.ones(x.size())
    # mask = mask.cuda() if x.is_cuda else mask
    # mask = F.grid_sample(mask, vgrid, align_corners=True)

    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1

    # return output * mask, mask

def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    # 等同于smurf 的 compute_occlusions_brox
    flow21_warped, _ = smurf_warp(flow21, flow12)
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()
    
def unsp_sequence_loss(flow_preds_forward, flow_preds_backward, img0, img1, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds_forward)    
    flow_loss = 0.0

    for i in range(n_predictions):
        i_loss = 0.0
        for ward in ["forward", "backward"]:
            if ward == "forward":
                warpped_img1, valid_warp_mask = smurf_warp(img1, flow_preds_forward[i])
                occlusion_mask = 1. - get_occu_mask_bidirection(flow_preds_forward[i], flow_preds_backward[i])
                mask_smurf = (occlusion_mask * valid_warp_mask).detach()
                i_loss += 1.0 * Census()(img0, warpped_img1, mask_smurf)
                i_loss += 4.0 * Second_Order_SmoothLoss()(img0, flow_preds_forward[i])
            else:
                warpped_img0, valid_warp_mask = smurf_warp(img0, flow_preds_backward[i])
                occlusion_mask = 1. - get_occu_mask_bidirection(flow_preds_backward[i], flow_preds_forward[i])
                mask_smurf = (occlusion_mask * valid_warp_mask).detach()
                i_loss += 1.0 * Census()(img1, warpped_img0, mask_smurf)
                i_loss += 4.0 * Second_Order_SmoothLoss()(img1, flow_preds_backward[i])
            # self-sup loss
        i_loss = i_loss / 2.

        i_weight = gamma**(n_predictions - i - 1)
        flow_loss += i_weight * i_loss

    return flow_loss

class ContrastMaximizationLoss(nn.Module):
    def __init__(self, losstype='var', blur_sigma=None, position='M', dt=64):
        super(ContrastMaximizationLoss, self).__init__()

        assert position == 'L' or position == 'R' or position == 'M'
        self.position = position
        self.dt = dt
        self.losstype = losstype
        self.default_blur=1.0
        self.blur_sigma = blur_sigma
        self.warp_module = Warp()

    def forward(self, flow: Tensor, events: Tensor):
        
        # TODO: 加高斯模糊，还没想好怎么写
        # blur_sigma=self.default_blur if self.blur_sigma is None else self.blur_sigma
        # if blur_sigma > 0:
        #     iwe = gaussian_filter(iwe, blur_sigma)
        B, C, H, W = events.shape    # B, 2*num_bins, h, w
        num_bins = C // 2
        event_voxel = events.reshape(B, 2, num_bins, H, W)
        # event_voxel = torch.sum(event_voxel, dim=1) # B, num_bins, h, w

        if self.position == 'M':
            event_warped_neg = event_warp_MID(event_voxel[:, 0, :, :, :], flow, dt=self.dt, num_bins=num_bins)
            event_warped_pos = event_warp_MID(event_voxel[:, 1, :, :, :], flow, dt=self.dt, num_bins=num_bins)
        else:   # self.position == 'L' or self.position == 'R':
            event_warped_neg = event_warp_LR(event_voxel[:, 0, :, :, :], flow, dt=self.dt, num_bins=num_bins, left_or_right=self.position)
            event_warped_pos = event_warp_LR(event_voxel[:, 1, :, :, :], flow, dt=self.dt, num_bins=num_bins, left_or_right=self.position)
 
        # event_warped_batchlist = warp_dense_events(self.warp_module, event_voxel_batch=events, flow_batch=flow)
        # B_num = len(event_warped_batchlist)

        loss = 0.0
        for i in range(B):
            event_warped_i = torch.cat([event_warped_neg[i], event_warped_pos[i]], dim=0)   # 2*num_bins, h, w
            event_warped_i = torch.cat([event_warped_neg[i], event_warped_pos[i]], dim=0)   # 2*num_bins, h, w
            event_image = torch.sum(event_warped_i, dim=0)     # h, w
            if self.losstype == 'sos':
                loss_e = - torch.mean(event_image*event_image)
            elif self.losstype == 'var':
                loss_e = - torch.var(event_image - torch.mean(event_image))
            loss += loss_e
        
        # for event_warped in event_warped_batchlist:
        #     if self.losstype == 'sos':
        #         loss_e = - torch.mean(event_warped*event_warped)
        #     elif self.losstype == 'var':
        #         loss_e = - torch.var(event_warped - torch.mean(event_warped))
        #     loss += loss_e
        return loss / B


def cmloss_spike(flow: Tensor, spike: Tensor, losstype='var', position='M', dt=128, weight=1.0, grad_filter=None, flowpgt=None):
    B, C, H, W = spike.shape    # B, 2*num_bins, h, w
    num_bins = C

    assert losstype == 'var' or losstype == 'sos', "losstype no included."

    if position == 'M':
        spike_warped = spike_warp_MID(spike, flow, dt=dt)
    else:   # self.position == 'L' or self.position == 'R':
        spike_warped = spike_warp_LR(spike, flow, dt=dt, left_or_right=position)
    # event_warped_batchlist = warp_dense_events(self.warp_module, event_voxel_batch=events, flow_batch=flow)
    # B_num = len(event_warped_batchlist)

    spike_image = torch.sum(spike_warped, dim=1, keepdim=True)     # B, 1, h, w
    grad_image = grad_filter(spike_image) # B, 1, H, W

    gt_grad = None
    if flowpgt is not None:
        if position == 'M':
            spike_gt = spike_warp_MID(spike, flowpgt, dt=dt)
        else:
            # print(flow.shape, flowpgt.shape)
            spike_gt = spike_warp_LR(spike, flowpgt, dt=dt, left_or_right=position)
        tfp_gt = torch.sum(spike_gt, dim=1, keepdim=True)     # B, 1, h, w
        gt_grad = grad_filter(tfp_gt) # B, 1, H, W

    loss = 0.0
    for i in range(B):
        img = grad_image[i].squeeze()     # h, w
        if losstype == 'sos':
            if flowpgt is not None:
                gt = gt_grad[i].squeeze()
                loss_e = (torch.mean(gt*gt) - torch.mean(img*img)) / torch.mean(gt*gt) # about 0 - 1
                loss_e = torch.clip(loss_e, 0, 1)
            else:
                loss_e = - torch.mean(img*img)
                loss_e = torch.clip(loss_e, 0, 1)
        elif losstype == 'var':
            if flowpgt is not None:
                gt = gt_grad[i].squeeze()
                loss_e = (torch.var(gt-torch.mean(gt)) - torch.var(img-torch.mean(img))) / torch.var(gt-torch.mean(gt))
                loss_e = torch.clip(loss_e, 0, 1)
            else:
                loss_e = - torch.var(img-torch.mean(img))
                loss_e = torch.clip(loss_e, 0, 1)
        loss += loss_e

    loss = weight * (loss / B)

    return loss


def cmloss_event(flow: Tensor, event: Tensor, losstype='var', position='M', dt=128, weight=1.0, max_loss=50):
    B, C, H, W = event.shape    # B, 2*num_bins, h, w
    num_bins = C

    assert losstype == 'var' or losstype == 'sos', "losstype no included."

    if position == 'M':
        event = event.reshape(B, 2, C//2, H, W)
        evtw_pos = event_warp_MID(event[:, 1, :, :, :], flow)
        evtw_pos = torch.sum(evtw_pos, dim=1)
        evtw_neg = event_warp_MID(event[:, 0, :, :, :], flow) # B, bins, H, W
        evtw_neg = torch.sum(evtw_neg, dim=1)
        event_warped = evtw_pos + evtw_neg  # B, H, W
    else:   # self.position == 'L' or self.position == 'R':
        event_warped = event_warp_LR(event, flow, dt=dt, left_or_right=position)
    # event_warped_batchlist = warp_dense_events(self.warp_module, event_voxel_batch=events, flow_batch=flow)
    # B_num = len(event_warped_batchlist)

    loss = 0.0
    for i in range(B):
        event_image = event_warped[i]
        if losstype == 'sos':
            loss_e = torch.clip(max_loss - torch.mean(event_image*event_image), 0, max_loss)
        elif losstype == 'var':
            loss_e = torch.clip(max_loss - torch.var(event_image - torch.mean(event_image)), 0 ,max_loss)
        loss += loss_e

    loss = weight * (loss / B)

    return loss


class ContrastMaximizationLossSpike(nn.Module):
    def __init__(self, losstype='var', blur_sigma=None, position='M', dt=64):
        super(ContrastMaximizationLossSpike, self).__init__()

        assert position == 'L' or position == 'R' or position == 'M'
        self.position = position
        self.dt = dt
        self.losstype = losstype
        self.default_blur=1.0
        self.blur_sigma = blur_sigma
        self.warp_module = Warp()

    def forward(self, flow: Tensor, spike: Tensor):
        
        # TODO: 加高斯模糊，还没想好怎么写
        # blur_sigma=self.default_blur if self.blur_sigma is None else self.blur_sigma
        # if blur_sigma > 0:
        #     iwe = gaussian_filter(iwe, blur_sigma)
        B, C, H, W = spike.shape    # B, 2*num_bins, h, w
        num_bins = C

        if self.position == 'M':
            spike_warped = spike_warp_MID(spike, flow, dt=64)
        else:   # self.position == 'L' or self.position == 'R':
            spike_warped = spike_warp_LR(spike, flow, dt=64, left_or_right=self.position)
        # event_warped_batchlist = warp_dense_events(self.warp_module, event_voxel_batch=events, flow_batch=flow)
        # B_num = len(event_warped_batchlist)

        loss = 0.0
        for i in range(B):
            spike_image = torch.sum(spike_warped, dim=0)     # h, w
            if self.losstype == 'sos':
                loss_e = - torch.mean(spike_image*spike_image)
            elif self.losstype == 'var':
                loss_e = - torch.var(spike_image - torch.mean(spike_image))
            loss += loss_e
    
        return loss / B
