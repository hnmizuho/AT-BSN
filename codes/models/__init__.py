import logging
logger = logging.getLogger('base')

def create_model(opt):
    model = opt['model']

    if model == "ATBSN":
        from .atbsn_model import ATBSN_Model as M
    elif model == "ATBSN_D":
        from .atbsn_d_model import ATBSN_D_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
