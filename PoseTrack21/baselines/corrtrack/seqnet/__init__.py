from .seqnet_api import SeqNetApi 
from .defaults import get_default_cfg

from torch.cuda import is_available 
from torch import load 


def get_seqnet_api(ckpt_path):
    cfg = get_default_cfg() 
    cfg.freeze() 

    model = SeqNetApi(cfg)
    if is_available():
        model.cuda() 

    ckpt = load(ckpt_path)
    model.load_state_dict(ckpt['model'], strict=False)

    print(f"loaded checkpoint {ckpt_path}")
    print(f"model was trained for {ckpt['epoch']} epochs")

    model.eval()
    return model

__all__ = ['SeqNetApi', 'get_default_cfg', 'get_seqnet_api']
