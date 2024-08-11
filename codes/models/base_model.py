import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.usage = opt['usage']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # # nameOfPatialLoad = "bsn" 其实等效于 network = self.MoBanNet.module.bsn，因此没必要。
    # def load_network(self, load_path, network, strict=True, nameOfPatialLoad=None):
    #     if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
    #         network = network.module

    #     if nameOfPatialLoad:
    #         part_network = getattr(network, nameOfPatialLoad)
    #     else:
    #         part_network = network

    #     load_net = torch.load(load_path)
    #     load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    #     for k, v in load_net.items():
    #         if k.startswith('module.'):
    #             load_net_clean[k[7:]] = v
    #         else:
    #             load_net_clean[k] = v

    #     if nameOfPatialLoad:
    #         load_net_clean_partial = OrderedDict()
    #         for k, v in load_net_clean.items():
    #             if k.startswith(nameOfPatialLoad):
    #                 load_net_clean_partial[k[len(nameOfPatialLoad):]] = v
    #     else:
    #         load_net_clean_partial = load_net_clean
        
    #     part_network.load_state_dict(load_net_clean_partial, strict=strict)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        
        network.load_state_dict(load_net_clean, strict=strict)

    # 这个用于将bsn的部分权重，加载到nbsn上。
    def load_network_bsn2nbsn(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if "nin_A" in k or "nin_B" in k or "nin_C" in k:
                continue
            if "dec1" in k and "conv1" in k:
                continue
            if "dec1" in k and "conv2" in k:
                continue
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        ori = network.state_dict()
        ori.update(load_net_clean)
        network.load_state_dict(ori, strict=strict)

    def save_training_state(self, epoch, iter_step, network_label, other_lable=None):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        if other_lable:
            save_filename = '{}_{}.state'.format(other_lable, network_label)
        else:
            save_filename = '{}_{}.state'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
