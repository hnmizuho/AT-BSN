import torch
import torch.nn as nn
import numpy as np
import os

def nor(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def baocun(filename=""):
    sz = os.path.getsize(filename)
    frame_sz = 1000 * 1024 // 8 #每一帧占多少字节
    data_len = sz // frame_sz #有多少帧
    frame_b = [] #按帧保存
    with open(filename, 'rb') as fu:
        for i in range(data_len):
            a = fu.read(frame_sz)
            frame_b.append(a) 
    print("len(frame_b)  ",len(frame_b))
    for i in range(0, len(frame_b)-41,5):
        data = frame_b[i:i+41]
        save_path = os.path.join("./bubble/","%s_%d.dat"%("1600",i))
        with open(save_path, "wb") as fr:
            for j in range(len(data)):
                fr.write(data[j])

def load_vidar_dat(filename, frame_cnt=None, width=640, height=480, reverse_spike=True):
    '''
    output: <class 'numpy.ndarray'> (frame_cnt, height, width) {0，1} float32
    '''
    array = np.fromfile(filename, dtype=np.uint8)

    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame

    spikes = []
    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(
                compr_frame, np.left_shift(1, b)), b))

        frame_ = np.stack(blist).transpose()
        frame_ = frame_.reshape((height, width), order='C')
        if reverse_spike:
            frame_ = np.flipud(frame_)
        spikes.append(frame_)

    return np.array(spikes).astype(np.float32)

def middleTFI(spike, middle, window=50):
    '''
    左右估计tfi
    input: <class 'numpy.ndarray'> (frame_cnt, h, w) {0，1} float32
    output: <class 'numpy.ndarray'> (h, w), (0,1) float32
    '''
    C, H, W = spike.shape
    lindex, rindex = np.zeros([H, W]), np.zeros([H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1): #往左包括自己50个,往右不包括自己也是50个
        l = l - 1
        if l>=0:
            newpos = spike[l, :, :]*(1 - np.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[r, :, :]*(1 - np.sign(rindex))
            distance = r*newpos
            rindex += distance
        if l<0 and r>=C:
            break

    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval 

    return tfi.astype(np.float32) #numpy.zeros 默认float64

def middleISI(spike, middle, window=50):
    '''
    左右估计tfi
    input: <class 'numpy.ndarray'> (frame_cnt, h, w) {0，1} float32
    output: <class 'numpy.ndarray'> (h, w), (0,1) float32
    '''
    C, H, W = spike.shape
    lindex, rindex = np.zeros([H, W]), np.zeros([H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1): #往左包括自己50个,往右不包括自己也是50个
        l = l - 1
        if l>=0:
            newpos = spike[l, :, :]*(1 - np.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[r, :, :]*(1 - np.sign(rindex))
            distance = r*newpos
            rindex += distance
        if l<0 and r>=C:
            break

    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    interval = nor(interval)

    return interval.astype(np.float32) #numpy.zeros 默认float64

def middleTFP(spike, middle, window=50):
    '''
    左右估计tfp
    input: <class 'numpy.ndarray'> (frame_cnt, h, w) {0，1} float32
    output: <class 'numpy.ndarray'> (h, w), (0,1) float32
    '''
    C, _, _ = spike.shape
    l,r = max(middle-window+1,0),min(middle+window+1,C)
    tfp = np.mean(spike[l:r],axis=0) #往左包括自己50个,往右不包括自己也是50个
    return tfp 


class torch_filter(nn.Module):
    def __init__(self, filter_weight, is_grad=False):
        super(torch_filter, self).__init__()
        assert type(filter_weight) == np.ndarray
        k=filter_weight.shape[0]
        filter=torch.tensor(filter_weight).unsqueeze(dim=0).unsqueeze(dim=0)
        # filters = torch.cat([filter, filter, filter], dim=0)

        self.conv = nn.Conv2d(1, 1, kernel_size=k,  bias=False, padding=int((k-1)/2))
        self.conv.weight.data.copy_(filter)
        self.conv.requires_grad_(is_grad)


    def forward(self,x):
        output = self.conv(x)
        output = torch.clip(output, 0, 1)
        return output


class GradFilter_Torch(nn.Module):
    def __init__(self, type='sobel', is_grad=False):
        super(GradFilter_Torch, self).__init__()
        # assert type(filter_weight) == np.ndarray
        if type == 'sobel':
            weight1 = np.array(
            [
                [-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1],
                ]
            )
            weight2 = np.array(
            [
                [1, 2, 1], 
                [0, 0, 0], 
                [-1, -2, -1],
                ]
            )
        elif type == 'scharr':
            weight1 = np.array(
            [
                [-3, 0, 3], 
                [-10, 0, 10], 
                [-3, 0, 3],
                ]
            )
            weight2 = np.array(
            [
                [3, 10, 3], 
                [0, 0, 0], 
                [-3, -10, -3],
                ]
            )
        
        k=weight1.shape[0]

        filter1=torch.tensor(weight1).unsqueeze(dim=0).unsqueeze(dim=0)
        filter2=torch.tensor(weight2).unsqueeze(dim=0).unsqueeze(dim=0)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=k,  bias=False, padding=int((k-1)/2))
        self.conv1.weight.data.copy_(filter1)
        self.conv1.requires_grad_(is_grad)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=k,  bias=False, padding=int((k-1)/2))
        self.conv2.weight.data.copy_(filter2)
        self.conv2.requires_grad_(is_grad)

    def forward(self,x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        # output = torch.clip(output, 0, 1)
        return output1+output2