import os
import math
import argparse
import random
import logging
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.data_sampler import DistIterSampler

import options.options as option
from utils import misc_utils, batch_utils
from dataset import create_dataloader, create_dataset
from models import create_model

# 去除了模型保存，状态保存。
# 去除了psnr，gt保存，val_log。

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, usage="test")

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print('Enabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        misc_utils.mkdirs((path for key, path in opt['path'].items() if key in [
            "experiments_root", "log", "test_images"
            ]))
            
        # config loggers. Before it, the log will not work
        misc_utils.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    else:
        misc_utils.setup_logger('base', opt['path']['log'], 'train_', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    misc_utils.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True # faster
    # torch.backends.cudnn.deterministic = True # reproducible

    #### create train and test dataloader
    dataset_ratio = 200  # enlarge the size of each epoch. whatever.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' or phase == 'val':
            pass
        elif phase == 'test':
            test_set = create_dataset(dataset_opt, phase)
            test_loader = create_dataloader(test_set, dataset_opt, phase, opt, None)
            if rank <= 0:
                logger.info('Number of test images in [{:s}]: {:d}'.format(
                    dataset_opt['d_name'], len(test_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert test_loader is not None

    #### create model
    model = create_model(opt)

    #### Testing
    logger.info('+++++++++++++++++++ Start Testing +++++++++++++++++++++')

    idx = 0
    for test_data in test_loader:
        idx += 1
        model.feed_data(test_data)
        model.test()

        visuals = model.get_current_visuals() #仍支持test时，用大batch_size。
        for tmp_i in range(len(test_data['img_path'])):

            img_name = os.path.splitext(os.path.basename(test_data['img_path'][tmp_i]))[0] #保留一个batch中的所有图片
            img_dir = os.path.join(opt['path']['test_images'], img_name)
            misc_utils.mkdir(img_dir)

            pred = misc_utils.tensor2img(visuals['pred'][tmp_i])  # --> uint8 0-255
            corruption = misc_utils.tensor2img(visuals['corruption'][tmp_i])  # --> uint8 0-255

            # Save pred images
            save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
            misc_utils.save_img(pred, save_img_path)

            # Save ground truth
            save_img_path_corruption = os.path.join(img_dir, '{:s}_corruption.png'.format(img_name))
            misc_utils.save_img(corruption, save_img_path_corruption)

    # log
    logger.info('# Test Finished. #')

if __name__ == '__main__':
    main()
