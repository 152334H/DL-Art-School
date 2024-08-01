import os
import pathlib
import sys
import time
import math

import scipy
import torch.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torchaudio
from audio2numpy import open_audio
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from shutil import get_terminal_size
import scp
import paramiko
from torch.utils.checkpoint import checkpoint
from torch import inf

import yaml

from trainer import networks

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

loaded_options = None

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################

# Conditionally uses torch's checkpoint functionality if it is enabled in the opt file.
def checkpoint(fn, *args):
    if loaded_options is None:
        enabled = False
    else:
        enabled = loaded_options['checkpointing_enabled'] if 'checkpointing_enabled' in loaded_options.keys() else True
    if enabled:
        return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)

def sequential_checkpoint(fn, partitions, *args):
    if loaded_options is None:
        enabled = False
    else:
        enabled = loaded_options['checkpointing_enabled'] if 'checkpointing_enabled' in loaded_options.keys() else True
    if enabled:
        return torch.utils.checkpoint.checkpoint_sequential(fn, partitions, *args)
    else:
        return fn(*args)

# A fancy alternative to if <flag> checkpoint() else <call>
def possible_checkpoint(opt_en, fn, *args):
    if loaded_options is None:
        enabled = False
    else:
        enabled = loaded_options['checkpointing_enabled'] if 'checkpointing_enabled' in loaded_options.keys() else True
    if enabled and opt_en:
        return torch.utils.checkpoint.checkpoint(fn, *args)
    else:
        return fn(*args)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def copy_files_to_server(host, user, password, files, remote_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    scpclient = scp.SCPClient(client.get_transport())
    scpclient.put(files, remote_path)

def get_files_from_server(host, user, password, remote_path, local_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password)
    scpclient = scp.SCPClient(client.get_transport())
    scpclient.get(remote_path, local_path)

####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


# Recursively detaches all tensors in a tree of lists, dicts and tuples and returns the same structure.
def recursively_detach(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    elif isinstance(v, list) or isinstance(v, tuple):
        out = [recursively_detach(i) for i in v]
        if isinstance(v, tuple):
            return tuple(out)
        return out
    elif isinstance(v, dict):
        out = {}
        for k, t in v.items():
            out[k] = recursively_detach(t)
        return out

def opt_get(opt, keys, default=None):
    assert not isinstance(keys, str)  # Common mistake, better to assert.
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def clip_grad_norm(parameters: list, parameter_names: list, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Equivalent to torch.nn.utils.clip_grad_norm_() but with the following changes:
    - Takes in a dictionary of parameters (from get_named_parameters()) instead of a list of parameters.
    - When NaN or inf norms are encountered, the parameter name is printed.
    - error_if_nonfinite removed.

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


Loader, Dumper = OrderedYaml()
def load_model_from_config(cfg_file=None, model_name=None, also_load_savepoint=True, load_path=None,
                           preloaded_options=None, strict_load=True, device=None):
    if preloaded_options is not None:
        opt = preloaded_options
    else:
        with open(cfg_file, mode='r') as f:
            opt = yaml.load(f, Loader=Loader)
    if model_name is None:
        model_cfg = opt['networks'].values()
        model_name = next(opt['networks'].keys())
    else:
        model_cfg = opt['networks'][model_name]
    if 'which_model_G' in model_cfg.keys() and 'which_model' not in model_cfg.keys():
        model_cfg['which_model'] = model_cfg['which_model_G']
    model = networks.create_model(opt, model_cfg).to(device)
    if also_load_savepoint and f'pretrain_model_{model_name}' in opt['path'].keys():
        assert load_path is None
        load_path = opt['path'][f'pretrain_model_{model_name}']
    if load_path is not None:
        print(f"Loading from {load_path}")
        sd = torch.load(load_path, map_location=device)
        model.load_state_dict(sd, strict=strict_load)
    return model


# Mapper for torch.load() that maps cuda devices to the correct CUDA device, but leaves CPU devices alone.
def map_cuda_to_correct_device(storage, loc):
    if str(loc).startswith('cuda'):
        return storage.cuda(torch.cuda.current_device())
    else:
        return storage.cpu()

def list_to_device(l, dev):
    return [anything_to_device(e, dev) for e in l]

def map_to_device(m, dev):
    return {k: anything_to_device(v, dev) for k,v in m.items()}
    
def anything_to_device(obj, dev):
    if isinstance(obj, list):
        return list_to_device(obj, dev)
    elif isinstance(obj, map):
        return map_to_device(obj, dev)
    elif isinstance(obj, torch.Tensor):
        return obj.to(dev)
    else:
        return obj


def ceil_multiple(base, multiple):
    """
    Returns the next closest multiple >= base.
    """
    res = base % multiple
    if res == 0:
        return base
    return base + (multiple - res)


def optimizer_to(opt, device):
    """
    Pushes the optimizer params from opt onto the specified device.
    """
    for param in opt.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#'''                       AUDIO UTILS                          '''
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def find_audio_files(base_path, globs=['*.wav', '*.mp3', '*.ogg', '*.flac']):
    path = pathlib.Path(base_path)
    paths = []
    for glob in globs:
        paths.extend([str(f) for f in path.rglob(glob)])
    return paths


def load_audio(audiopath, sampling_rate, raw_data=None):
    audiopath = str(audiopath)
    if raw_data is not None:
        # Assume the data is wav format. SciPy's reader can read raw WAV data from a BytesIO wrapper.
        audio, lsr = load_wav_to_torch(raw_data)
    else:
        if audiopath[-4:] == '.wav':
            audio, lsr = load_wav_to_torch(audiopath)
        elif audiopath[-5:] == '.flac':
            import soundfile as sf
            audio, lsr = sf.read(audiopath)
            audio = torch.FloatTensor(audio)
        elif audiopath[-4:] == '.aac':
            # Process AAC files using pydub. I'd use this for everything except I'm cornered into backwards compatibility.
            from pydub import AudioSegment
            asg = AudioSegment.from_file(audiopath)
            dtype = getattr(np, "int{:d}".format(asg.sample_width * 8))
            arr = np.ndarray((int(asg.frame_count()), asg.channels), buffer=asg.raw_data, dtype=dtype)
            arr = arr.astype('float') / (2 ** (asg.sample_width * 8 - 1))
            arr = arr[:,0]
            audio = torch.FloatTensor(arr)
            lsr = asg.frame_rate
        else:
            audio, lsr = open_audio(audiopath)
            audio = torch.FloatTensor(audio)

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio


def pad_or_truncate(t, length):
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]


def load_wav_to_torch(full_path):
    import scipy.io.wavfile
    sampling_rate, data = scipy.io.wavfile.read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def get_network_description(network):
    """Get the string and total parameters of the network"""
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    return str(network), sum(map(lambda x: x.numel(), network.parameters()))


def print_network(net, name='some network'):
    s, n = get_network_description(net)
    net_struc_str = '{}'.format(net.__class__.__name__)
    print('Network {} structure: {}, with parameters: {:,d}'.format(name, net_struc_str, n))
    print(s)
