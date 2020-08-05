import os
import io
from glob import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import PIL.Image


# Note! This is l2 square, not l2
def l2(a, b):
    return torch.pow(torch.abs(a - b), 2).sum(dim=1)


# required when we load optimizer from a checkpoint
def optimizer_cuda(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_ckpt_path(base_dir, ckpt_num):
    if ckpt_num is None:
        return get_recent_ckpt_path(base_dir)
    files = glob(os.path.join(base_dir, "*.pt"))
    for f in files:
        if 'ckpt_%08d.pt' % ckpt_num in f:
            return f, ckpt_num
    raise Exception("Did not find ckpt_%s.pt" % ckpt_num)


def get_recent_ckpt_path(base_dir):
    files = glob(os.path.join(base_dir, "*.pt"))
    files.sort()
    if len(files) == 0:
        return None, None
    max_step = max([f.rsplit('_', 1)[-1].split('.')[0] for f in files])
    paths = [f for f in files if max_step in f]
    if len(paths) == 1:
        return paths[0], int(max_step)
    else:
        raise Exception("Multiple most recent ckpts %s" % paths)


def image_grid(image, n=4):
    return vutils.make_grid(image[:n], nrow=n).cpu().detach().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def slice_tensor(input, indices):
    ret = {}
    for k, v in input.items():
        ret[k] = v[indices]
    return ret

def ensure_shared_grads(model, shared_model):
    """for A3C"""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def compute_gradient_norm(model):
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += (p.grad.data ** 2).sum().item()
    return grad_norm

def compute_weight_norm(model):
    weight_norm = 0
    for p in model.parameters():
        if p.data is not None:
            weight_norm += (p.data ** 2).sum().item()
    return weight_norm


def compute_weight_sum(model):
    weight_sum = 0
    for p in model.parameters():
        if p.data is not None:
            weight_sum += p.data.abs().sum().item()
    return weight_sum

def fig2tensor(draw_func):
    def decorate(*args, **kwargs):
        tmp = io.BytesIO()
        fig = draw_func(*args, **kwargs)
        fig.savefig(tmp, dpi=88)
        tmp.seek(0)
        fig.clf()
        return TF.to_tensor(PIL.Image.open(tmp))

    return decorate


def tensor2np(t):
    if isinstance(t, torch.Tensor):
        return t.clone().detach().cpu().numpy()
    else:
        return t


def tensor2img(tensor):
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor.squeeze(0)
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    import cv2
    cv2.imwrite('tensor.png', img)


def obs2tensor(obs, device):
    if isinstance(obs, list):
        obs = list2dict(obs)

    return OrderedDict([
        (k, torch.tensor(np.stack(v), dtype=torch.float32).to(device)) for k, v in obs.items()
    ])


# transfer a numpy array into a tensor
def to_tensor(x, device):
    if isinstance(x, dict):
        return OrderedDict([
            (k, torch.as_tensor(v, dtype=torch.float32).to(device))
            for k, v in x.items()])
    if isinstance(x, list):
        return [torch.as_tensor(v, dtype=torch.float32).to(device)
                for v in x]
    return torch.as_tensor(x, dtype=torch.float32).to(device)


def list2dict(rollout):
    ret = OrderedDict()
    for k in rollout[0].keys():
        ret[k] = []
    for transition in rollout:
        for k, v in transition.items():
            ret[k].append(v)
    return ret


# From softlearning repo
def flatten(unflattened, parent_key='', separator='/'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return OrderedDict(items)


# From softlearning repo
def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result
