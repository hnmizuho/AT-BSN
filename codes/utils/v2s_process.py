import os
import cv2
import numpy as np
from tqdm import tqdm

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def SpikeToRaw(SpikeSeq, save_path):
    """
        SpikeSeq: Numpy array (sfn x h x w)
        save_path: full saving path (string)
        Rui Zhao
    """
    sfn, h, w = SpikeSeq.shape
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        # 模拟相机的倒像
        spike = np.flipud(SpikeSeq[img_id, :, :])
        # numpy按自动按行排，数据也是按行存的
        spike = spike.flatten()
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())

    fid.close()

    return

def v2s_interface(datafolder='F:\\tmp\\citystreet01', savefolder=None, threshold=5.0):
    filelist = sorted(os.listdir(datafolder))
    datas = [fn for fn in filelist if fn.endswith("png")]
    T = len(datas)
    frame0 = cv2.imread(os.path.join(datafolder, datas[0]))
    # frame0[..., 0:2] = 0
    # cv2.imshow('red', frame0)
    # cv2.waitKey(0)
    H, W, C = frame0.shape
    # exit(0)

    spikematrix = np.zeros([T, H//2, W//2], np.uint8)
    # integral = np.array(frame0gray).astype(np.float32)
    integral = np.random.random(size=([H//2,W//2])) * threshold
    Thr = np.ones_like(integral).astype(np.float32) * threshold

    for t in range(0, T):
        # print('spike frame %s' % datas[t])
        frame = cv2.imread(os.path.join(datafolder, datas[t]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gray = gray / 255.0
        integral += gray
        fire = (integral - Thr) >= 0
        fire_pos = fire.nonzero()
        
        integral[fire_pos] -= threshold
        # integral[fire_pos] = 0.0
        spikematrix[t][fire_pos] = 1

    if savefolder:
        np.save(os.path.join(savefolder, "spike_debug.npy"), spikematrix)

    return spikematrix

def x4k1000fps_proc(rootfolder, save_rootfolder):
    # rootfolder = 'F:/Downloads/XVFI1000FPS/train'
    # save_rootfolder = 'F:/Downloads/XVFI1000FPS/train_spike'
    # print(os.path.join(save_rootfolder, '\002\occ008.320_f2881'.replace('\\', '/')))
    check_folder(save_rootfolder)
    l0_folders = os.listdir(rootfolder)
    l0_folders = [os.path.join(rootfolder, p) for p in l0_folders]  # 0-172
    print(l0_folders[0])
    l1_folders = {}
    for l0_folder in l0_folders:
        l1_folders[l0_folder] = [os.path.join(l0_folder, p) for p in os.listdir(l0_folder)] # 0-172 / occ.1211212
    # sumnum = 0
    for k,v in l1_folders.items():
        # sumnum += len(v)
        # print(k)
        for datafolder in tqdm(v):
            # print(save_rootfolder, datafolder, datafolder.replace(rootfolder, ''))
            save_datafolder = save_rootfolder + datafolder.replace(rootfolder, '').replace('\\','/')
            print(save_datafolder)
            check_folder(save_datafolder)
            
            s = v2s_interface(datafolder, savefolder=None, threshold=2.0)
            print(s.shape)
            # exit(0)
            SpikeToRaw(s, os.path.join(save_datafolder,'spike.dat'))


if __name__ == "__main__":
    rootfolder = '/home/data/jyzhang/XVFI/train_fullRGB'
    savefolder = '/home/data/jyzhang/XVFI/train_spike_2xds'

    x4k1000fps_proc(rootfolder, savefolder)