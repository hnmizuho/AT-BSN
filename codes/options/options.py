import os
import os.path as osp
import logging
import yaml
from utils.misc_utils import OrderedYaml, get_timestamp
Loader, Dumper = OrderedYaml()


def parse(opt_path, usage="train"):
    assert usage in ["train", "val", "test"], "Usage [%s] is not recognized." % usage

    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['usage'] = usage

    # replace str type "~" to /home/username
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)) # pardir means parent dir. root is in the same dir as README.
    if usage == "train":
        opt["expr_rootname"] = get_timestamp() + "_" + opt['name']
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt["expr_rootname"])
    elif usage == "val":
        opt["expr_rootname"] = get_timestamp() + "_VALIDATION_" + opt['name']
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt["expr_rootname"])
    elif usage == "test":
        opt["expr_rootname"] = get_timestamp() + "_TESTING_" + opt['name']
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt["expr_rootname"])

    opt['path']['experiments_root'] = experiments_root
    opt['path']['models'] = osp.join(experiments_root, 'models')
    opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
    opt['path']['log'] = experiments_root
    if usage == "test":
        opt['path']['test_images'] = osp.join(experiments_root, 'test_images')
    else:
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

    # change some options for debug mode
    if usage == "train" and 'debug' in opt['name'].lower():
        opt['train']['niter'] = 16
        opt['train']['val_freq'] = 8
        opt['logger']['print_freq'] = 1
        opt['logger']['save_checkpoint_freq'] = 99999

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    assert opt["path"]["pretrain_model"] is not None, "ERROR: pretrain_model path is None, but resume_state is not None."
    assert os.path.split(os.path.split(opt["path"]["pretrain_model"])[0])[0] == os.path.split(os.path.split(opt["path"]["resume_state"])[0])[0], "ERROR: pretrain_model path is not the same as resume_state path."
    assert os.path.split(opt["path"]["pretrain_model"])[1].split(".")[0] == os.path.split(opt["path"]["resume_state"])[1].split(".")[0], "ERROR: pretrain_model name is not the same as resume_state name."

    opt["expr_rootname"] = get_timestamp() + "_RESUME_" + opt['name']
    experiments_root = osp.join(opt['path']['root'], 'experiments', opt["expr_rootname"])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['models'] = osp.join(experiments_root, 'models')
    opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
    opt['path']['log'] = experiments_root
    opt['path']['val_images'] = osp.join(experiments_root, 'val_images')
