import copy
import numpy as np
import torch
from torch import nn
import warnings
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings('ignore')

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



def select_users_by_value(args, bandwidth, g, d):
    g = np.array(g)
    selection = np.zeros(args.num_users, dtype=int)

    # Sort users by g descending
    sorted_indices = np.argsort(-g)
    
    total = 0
    for idx in sorted_indices:
        if total + d[idx] <= bandwidth:
            selection[idx] = 1
            total += d[idx]
        else:
            break

    return selection, g[idx]