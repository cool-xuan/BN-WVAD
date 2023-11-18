import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("AUC: {:.4f}\n".format(test_info["AUC"][-1]))
    fo.write("AP: {:.4f}\n".format(test_info["AP"][-1]))
