import argparse
import os

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--len_feature', type = int, default = 1024)

    parser.add_argument('--root_dir', type = str, default = 'xd/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    
    parser.add_argument('--model_path', type = str, default = 'ckpts/')
    parser.add_argument('--lr', type = str, default = '[0.0001]*1000', help = 'learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 64)
    
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--num_segments', type = int, default = 200)
    parser.add_argument('--seed', type = int, default = 2022, help = 'random seed (-1 for no manual seed)')
    
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--processed', action = 'store_true', help = 'is process training data to [segments x c]')
    parser.add_argument('--plot_freq', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    
    parser.add_argument('--version', type=str, default='train')
    
    parser.add_argument('--ratio_sample', type=float, default=0.2)
    parser.add_argument('--ratio_batch', type=float, default=0.4)
    
    parser.add_argument('--ratios', type=int, nargs='+', default = [16, 32])
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default = [1, 1, 1])
    
    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    args.lr  = eval(args.lr)
    args.num_iters = len(args.lr)

    return args
