# Default Python Pacvkages
import math
from time import time
from typing import Dict, Any
import json
import hashlib
import logging
import os
import datetime 

# Standard Python Packages.
import torch
from tqdm import tqdm
import numpy as np
from munch import Munch

# Project Specific Dependencies

def timer_at_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def convert_parsed_args_to_config(parser, parsed_args):
    """ convert parsed Args object to Munch structure
    Args:
        parser (_type_): argparse parser
        parsed_args (_type_): parser with parsed argument string

    Returns:
        Munch: Dictionary like structure that can be accessed like an object
    """
    # Process Parsed arguments
    config = Munch()
    config.arg_gnames = []
    title_dict = {"positional arguments": "posArgs", "optional arguments":"optArgs"}
    title_map = lambda t: title_dict[t] if t in title_dict else t
    for group in parser._action_groups:
        title = title_map(group.title)
        config.arg_gnames.append(title)
        config[title]=Munch({a.dest:getattr(parsed_args,a.dest,None) for a in group._group_actions})

    return config

import socket

def in_server():
    host_name = socket.gethostname().split(".")[0]
    return "elim" not in host_name


def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except:
        return False

def init_dac_logger(name: str =None, base_path: str=None):
    if base_path is None:
        env_log_path = os.getenv("DAC_LOG_DIR")
        base_path = env_log_path if env_log_path else "./"

    if name is None:
        now = datetime.datetime.now()
        name = "DefaultDacLogger " + now.strftime('%Y-%m-%d T%H:%M:%S')

    logger = init_logger(name, base_path)
    return logger

def init_logger(name: str, base_path: str):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    file_path = os.path.join(base_path, name + '.log')

    logger = logging.getLogger(name)
    logging.getLogger().handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler(file_path, mode='w')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    return logger


def has_attributes(obj, list_of_attributes): 
    return all([hasattr(obj, a) for a in list_of_attributes])

def iter_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def verbose_iterator(it, vb): return tqdm(it) if vb else it

def conv2tensor(l):
    if isinstance(l[0], torch.Tensor):
        return torch.stack(l)
    else:
        return torch.tensor(l)

def v_iter(iterator, verbose = True, label=""):
    """
    Returns a verbose iterator i.e. tqdm enabled iterator if verbose is True. 
    It also attaches the passed message to the iterator.
    """
    if verbose:
        vb_iterator = tqdm(iterator)
        vb_iterator.set_description(label)
    else:
        vb_iterator = iterator

    return vb_iterator

def v_map(fxn, iterable, batch_size = 1, reduce_fxn = lambda x:x, verbose = True, label = ""):
    it = v_iter(iter_batch(iterable, batch_size), verbose, label)
    return reduce_fxn(list(map(fxn, it)))

# def torch_map_then_flatten(fxn, iterator, batch_size=None, verbose = False, label=""):
#     if batch_size is None:
#         return fxn(iterator)
#     else:
#         return 

    
def get_eucledian_dist(s1, s2):
    s1, s2 = np.array(s1), np.array(s2)
    return np.linalg.norm(s1 - s2)


def cmap(fxn, iterable):
    return list(map(fxn, iterable))


def get_one_hot_list(tt_size):
    """
    Get list of one hot vectors. 
    input - tt_size: Size of the list. 
    Return - list of One Hot Vectors. 
    """
    zero_matrix = torch.zeros(
        (tt_size, tt_size), dtype=torch.float32, device="cpu")
    tt_tensor = zero_matrix.scatter_(1, torch.LongTensor(
        range(tt_size)).unsqueeze(1), 1).numpy()
    return [tuple(tt) for tt in tt_tensor]

def dict_hash(d: Dict[str, Any]) -> str:
    """returns an MD5 hash of a dictionary."""
    dhash = hashlib.sha1()

    if isinstance(d, dict):
        d = {str(k): str(d[k]) for k in list(d)[::math.ceil(len(d) / 1000)]}
        encoded = json.dumps(d).encode()

    elif isinstance(d, list):
        encoded = str(d[::math.ceil(len(d) / 1000)]).encode()

    else:
        assert False, "Type not defined for encoding "

    dhash.update(encoded)
    return dhash.hexdigest()

def matrix_hash(m):
    if isinstance(m,str):
        return m
    if isinstance(m,torch.Tensor):
        m = m.reshape(-1).clone().detach().numpy().astype("float32").tolist() 
        return tuple(m)
    else:
        m = torch.tensor(m).reshape(-1).clone().detach().numpy().astype("float32").tolist() 
        return tuple(m)


def plot_distributions_as_rgb_array(dists:dict)->np.ndarray:
    import matplotlib.pyplot as plt
    import math 

    n_plots = len(dists)
    n_row = math.ceil(math.sqrt(n_plots))
    n_col = math.ceil(n_plots/n_row)

    fig, axs = plt.subplots(n_row, n_col)
    fig.suptitle('MDP Distributions')
    fig.set_size_inches(n_col*10,n_row*4)

    try:
        dist_items = list(dists.items())
        for i in range(n_row):
            for j in range(n_col):
                label, dist = dist_items[i*n_col + j]
                axs[i,j].set_title(label)
                axs[i,j].hist(dist, bins = 100)
    except:
        pass

    plt.show()
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return rgb_array

def plot_distribution_as_rgb_array(dist:np.ndarray)->np.ndarray:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(figsize = (10,4),  dpi=200)
    axs.hist(dist, bins = 100)

    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb_array = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return rgb_array



def write_gif(episode_images, metrics, metric_metas, gif_path, scaling = 0.75,  save_mp4=True):
    """
    metrics = list of metric array 
    labels = list of x and y lavels for each metric array
    """
    episode_len = len(episode_images)
    assert all([len(episode_images) == len(m) for m in metrics])
    
    import plotly.graph_objects as go
    from io import BytesIO
    from PIL import Image
    import numpy as np
    
    rep_figs, rep_imgs= [], [None]*len(metrics)
    for i, m, meta in zip(range(len(metrics)), metrics, metric_metas): 
        rep_figs.append(go.Figure(data=go.Scatter(x=[], y=[])))
        rep_figs[-1].update_layout(
            title=meta["title"],
            xaxis_title=meta["xaxis_title"],
            yaxis_title=meta["yaxis_title"],
        )

    episode_stats = []
    _obs = Image.fromarray(episode_images[0])
    width, height = int(scaling * _obs.width), int(scaling * _obs.height)
    step_i = 0


    while step_i < len(episode_images):
        # update figure for all metrics
        for i, m, meta in zip(range(len(metrics)), metrics, metric_metas):
            rep_figs[i]['data'][0]['x'] += tuple([step_i])
            rep_figs[i]['data'][0]['y'] += tuple([m[step_i]])

            rep_imgs[i] = Image.open(BytesIO(rep_figs[i].to_image(format="png",
                                 width=width, height=height)))

        # obs
        obs = Image.fromarray(episode_images[step_i])
        obs = obs.resize((width, height), Image.ANTIALIAS)

        # combine repeat image + actual obs + score image
        all_widths = [obs.width , *[img.width for img in rep_imgs]]
        overall_width = np.sum(all_widths)
        overall_img = Image.new('RGB', (overall_width, height))
        overall_img.paste(obs, (0, 0))
        rep_img_start_positions = np.cumsum(all_widths)
                              
        for rep_img, pos in zip(rep_imgs,rep_img_start_positions[:-1]):                     
            overall_img.paste(rep_img, (pos, 0))
            episode_stats.append(overall_img)

        # incr counters
        step_i += 1

    assert step_i == len(episode_images)

    # save as gif
    episode_stats[0].save(gif_path, save_all=True, append_images=episode_stats[1:], optimize=False, loop=1)

    # save video
    if save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(gif_path)
        clip.write_videofile(gif_path.replace('.gif', '.mp4'))

    return gif_path
