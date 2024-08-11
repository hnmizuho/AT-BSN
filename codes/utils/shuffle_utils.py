import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from einops.layers.torch import Rearrange
import numpy as np

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

def gen_cylic_idx(x):
    if x == 1:
        return [0]
    idx_matrix = np.arange(x * x).reshape(x, x)
    abso_center = (x**2)//2
    def foo(x, idx_matrix):
        idx = [[-1] * x for _ in range(x)]
        if np.random.choice([0,1], 1)[0]:
            #clockwise
            for i in range(x-1):
                idx[i][0] = idx_matrix[i+1][0]
                idx[0][i+1] = idx_matrix[0][i]
                idx[i+1][-1] = idx_matrix[i][-1]
                idx[-1][i] = idx_matrix[-1][i+1]
        else:
            #anti-clockwise
            for i in range(x-1):
                idx[0][i] = idx_matrix[0][i+1]
                idx[i+1][0] = idx_matrix[i][0]
                idx[-1][i+1] = idx_matrix[-1][i]
                idx[i][-1] = idx_matrix[i+1][-1]
        if x==2:
            return idx
        if x==3:
            tmp = np.random.choice([0,1,2,3,5,6,7,8], 1)[0]
            idx[1][1], idx[tmp//3][tmp%3] = idx[tmp//3][tmp%3], abso_center
            return idx
        else:
            inner_idx = foo(x-2, idx_matrix[1:-1, 1:-1])
            for i in range(x-2):
                for j in range(x-2):
                    idx[i+1][j+1] = inner_idx[i][j]
            return idx
    idx = foo(x, idx_matrix)
    idx = np.array(idx).reshape(-1)
    return idx

def cylic_pixelshuffle_downsampling(img, pd_size, pd):
    b,c,h,w = img.shape
    if h % pd_size != 0:
        img = F.pad(img, (0, 0, 0, pd_size - h%pd_size), mode='constant', value=0)
    if w % pd_size != 0:
        img = F.pad(img, (0, pd_size - w%pd_size, 0, 0), mode='constant', value=0)

    pd_x = pixel_shuffle_down_sampling(img, pd_size, pd)

    re_pdx = Rearrange('b c (p1 h) (p2 w) -> (b c h w) (p1 p2)', p1=pd_size, p2=pd_size)(pd_x) # 不会影响pdx。pdx不变。
    # re_idx = torch.from_numpy(gen_cylic_idx(pd_size)).to(img.device)
    re_idx = gen_cylic_idx(pd_size)
    cylic_shuffle_pdx = re_pdx[:,re_idx]
    cylic_shuffle_pdx = Rearrange('(b c h w) (p1 p2) -> b c (p1 h) (p2 w)', p1=pd_size, p2=pd_size, b=b, c=c, h=pd_x.shape[2]//pd_size, w=pd_x.shape[3]//pd_size)(cylic_shuffle_pdx)

    return pd_x, cylic_shuffle_pdx

def rand_pixelshuffle_downsampling_old(img, pd_size, pd):
    b,c,h,w = img.shape
    if h % pd_size != 0:
        img = F.pad(img, (0, 0, 0, pd_size - h%pd_size), mode='constant', value=0)
    if w % pd_size != 0:
        img = F.pad(img, (0, pd_size - w%pd_size, 0, 0), mode='constant', value=0)
    new_h, new_w = img.shape[2:]

    re_img2 = Rearrange('b c (h p1) (w p2) -> (b c h w) (p1 p2)', p1=pd_size, p2=pd_size)(img)
    idx = torch.stack([torch.randperm(re_img2.shape[1]) for _ in range(re_img2.shape[0])], dim=0).to(img.device)
    # idx = torch.stack([torch.randperm(re_img2.shape[1]) for _ in range(re_img2.shape[0])], dim=0)
    shuffle_img2 = re_img2[torch.arange(re_img2.shape[0]).unsqueeze(1), idx]
    shuffle_img2 = Rearrange('(b c h w) (p1 p2) -> b c (h p1) (w p2)', p1=pd_size, p2=pd_size, b=b, c=c, h=new_h//pd_size, w=new_w//pd_size)(shuffle_img2)

    pd_shuffle_x2 = pixel_shuffle_down_sampling(shuffle_img2, pd_size, pd)

    return pd_shuffle_x2, idx

def gen_permutation(n):
    return np.array([np.random.permutation(n) for _ in range(n)], dtype=np.int64)

def rand_pixelshuffle_downsampling_new(img, pd_size, pd, pd_permutation):
    b,c,h,w = img.shape
    if h % pd_size != 0:
        img = F.pad(img, (0, 0, 0, pd_size - h%pd_size), mode='constant', value=0)
    if w % pd_size != 0:
        img = F.pad(img, (0, pd_size - w%pd_size, 0, 0), mode='constant', value=0)
    new_h, new_w = img.shape[2:]

    re_img2 = Rearrange('b c (h p1) (w p2) -> (b c) (h w) (p1 p2)', p1=pd_size, p2=pd_size)(img)
    idx = torch.from_numpy(pd_permutation[np.random.randint(pd_size**2, size=re_img2.shape[1])]).to(img.device)
    shuffle_img2 = re_img2[:,torch.arange(re_img2.shape[1]).unsqueeze(1), idx]
    shuffle_img2 = Rearrange('(b c) (h w) (p1 p2) -> b c (h p1) (w p2)', p1=pd_size, p2=pd_size, b=b, c=c, h=new_h//pd_size, w=new_w//pd_size)(shuffle_img2)

    pd_shuffle_x2 = pixel_shuffle_down_sampling(shuffle_img2, pd_size, pd)

    return pd_shuffle_x2, idx

def rand_pixelshuffle_upsampling_old(down_img, pd_size, pd, idx):
    shuffle_x2 = pixel_shuffle_up_sampling(down_img, pd_size, pd)
    b,c,new_h,new_w = shuffle_x2.shape

    de_img = Rearrange('b c (h p1) (w p2) -> (b c h w) (p1 p2)', p1=pd_size, p2=pd_size)(shuffle_x2)
    de_idx = idx.argsort(dim=1)
    deshuffle_img2 = de_img[torch.arange(de_img.shape[0]).unsqueeze(1), de_idx]
    deshuffle_img2 = Rearrange('(b c h w) (p1 p2) -> b c (h p1) (w p2)', p1=pd_size, p2=pd_size, b=b, c=c, h=new_h//pd_size, w=new_w//pd_size)(deshuffle_img2)
    return deshuffle_img2

def rand_pixelshuffle_upsampling_new(down_img, pd_size, pd, idx):
    shuffle_x2 = pixel_shuffle_up_sampling(down_img, pd_size, pd)
    b,c,new_h,new_w = shuffle_x2.shape

    de_img = Rearrange('b c (h p1) (w p2) -> (b c) (h w) (p1 p2)', p1=pd_size, p2=pd_size)(shuffle_x2)
    de_idx = idx.argsort(dim=1)
    deshuffle_img2 = de_img[:,torch.arange(de_img.shape[1]).unsqueeze(1), de_idx]
    deshuffle_img2 = Rearrange('(b c) (h w) (p1 p2) -> b c (h p1) (w p2)', p1=pd_size, p2=pd_size, b=b, c=c, h=new_h//pd_size, w=new_w//pd_size)(deshuffle_img2)
    return deshuffle_img2

def R3_postprocess(noisy, pred, network, R3_T, R3_P, valid_pd_size, pd_pad):
    b,c,h,w = noisy.shape
    if h % valid_pd_size != 0:
        noisy = F.pad(noisy, (0, 0, 0, valid_pd_size - h%valid_pd_size), mode='constant', value=0)
    if w % valid_pd_size != 0:
        noisy = F.pad(noisy, (0, valid_pd_size - w%valid_pd_size, 0, 0), mode='constant', value=0)

    denoised = torch.empty(*(noisy.shape), R3_T, device=noisy.device)
    for t in range(R3_T):
        indice = torch.rand_like(noisy)
        mask = indice < R3_P

        tmp_input = torch.clone(pred).detach()
        tmp_input[mask] = noisy[mask]
        p = pd_pad
        tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
        if pd_pad == 0:
            denoised[..., t] = network(tmp_input)
        else:
            denoised[..., t] = network(tmp_input)[:,:,p:-p,p:-p]

    return torch.mean(denoised, dim=-1)


# def softening(img, temperature):
#     img = img/temperature
#     b,c,h,w = img.shape
#     img = img.view(b*c,h*w)
#     img_sum = torch.sum(torch.exp(img), dim=1).unsqueeze(1)
#     img = torch.exp(img)/img_sum
#     img = img.view(b,c,h,w)
#     return img, img_sum

# def de_softening(img, img_sum, temperature):
#     b,c,h,w = img.shape
#     img = img.view(b*c,h*w)
#     img = torch.log(img*img_sum)*temperature
#     img = img.view(b,c,h,w)
#     return img

def softening(img, temperature):
    img = torch.exp(-img-temperature)
    return img

def de_softening(img, temperature):
    img = -torch.log(img)-temperature
    return img

####################### neighbor2neighbor 
operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

################## blind2unblind
def depth_to_space_b2ub(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)
def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=torch.int64,
                       device=img.device)
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                         dtype=torch.int64,
                         device=img.device)

    if mask_type == 'random':
        torch.randint(low=0,
                      high=len(idx_list),
                      size=(n * h // width * w // width, ),
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)
    elif mask_type == 'batch':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=torch.int64,
                                device=img.device)

    mask[rd_pair_idx] = 1

    mask = depth_to_space_b2ub(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv
class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks
    
def pixel_unshuffle_sdap(input, factor):
    """
    (n, c, h, w) ===> (n*factor^2, c, h/factor, w/factor)
    """
    if factor == 1:
        return input
    
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor,
        out_width, factor)

    batch_size *= factor ** 2
    unshuffle_out = input_view.permute(0, 3, 5, 1, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

def pixel_shuffle_sdap(input, factor):
    """
    (n*factor^2, c, h/factor, w/factor) ===> (n, c, h, w)
    """
    if factor == 1:
        return input
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height * factor
    out_width = in_width * factor

    batch_size /= factor ** 2
    batch_size = int(batch_size)
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height,
        in_width)

    unshuffle_out = input_view.permute(0, 3, 4, 1, 5, 2).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

def rand_pixelshuffle_downsampling_batch(img, pd_size, pd, pd_permutation):
    b,c,h,w = img.shape
    if h % pd_size != 0:
        img = F.pad(img, (0, 0, 0, pd_size - h%pd_size), mode='constant', value=0)
    if w % pd_size != 0:
        img = F.pad(img, (0, pd_size - w%pd_size, 0, 0), mode='constant', value=0)
    new_h, new_w = img.shape[2:]

    re_img2 = Rearrange('b c (h p1) (w p2) -> (b c) (h w) (p1 p2)', p1=pd_size, p2=pd_size)(img)
    idx = torch.from_numpy(pd_permutation[np.random.randint(pd_size**2, size=re_img2.shape[1])]).to(img.device)
    shuffle_img2 = re_img2[:,torch.arange(re_img2.shape[1]).unsqueeze(1), idx]
    shuffle_img2 = Rearrange('(b c) (h w) (p1 p2) -> (b p1 p2) c h w', p1=pd_size, p2=pd_size, b=b, c=c, h=new_h//pd_size, w=new_w//pd_size)(shuffle_img2)

    return shuffle_img2, idx

def rand_pixelshuffle_upsampling_batch(down_img, pd_size, pd, idx):
    b,c,h,w = down_img.shape
    b = b//pd_size**2

    de_img = Rearrange('(b p1 p2) c h w -> (b c) (h w) (p1 p2)', p1=pd_size, p2=pd_size)(down_img)
    de_idx = idx.argsort(dim=1)
    deshuffle_img2 = de_img[:,torch.arange(de_img.shape[1]).unsqueeze(1), de_idx]
    deshuffle_img2 = Rearrange('(b c) (h w) (p1 p2) -> b c (h p1) (w p2)', p1=pd_size, p2=pd_size, b=b, c=c, h=h, w=w)(deshuffle_img2)
    return deshuffle_img2