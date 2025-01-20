from MLVAE.main import main as train_MLVAE
from CLDM.main import main as train_CLDM
from CLDM.sample import main as sample_CLDM
from CLDM.evaluation import main as eval_CLDM
from data.dataprocessing import Default_processing as split_data
from data.dataprocessing import data_syn as syn_data

import argparse
import importlib

def execute_function(method, mode):

    main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='abalone_15', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: syn, split, train, sample, or eval .')
    parser.add_argument('--method', type=str, default='MLVAE', help='Method: data, MLVAE or CLDM.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--exp', type=int, default=0, help='data index.')
    parser.add_argument('--threshold', type=float, default=0.2, help='threshold of drop rate for conditional information')

    # configs for the training of MLVAE
    parser.add_argument('--dist', type=float, default=1,
                        help='the param for dist_loss.')
    parser.add_argument('--proto', type=float, default=1,
                        help='the param for loss_pt.')
    parser.add_argument('--kld', type=float, default=1,
                        help='the param for loss_kld.')
    parser.add_argument('--condition', action='store_false')

    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')

    # configs for sampling
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')

    # configs for artificial data
    parser.add_argument('--means', type=float, default=0.5, help='means of minority')
    parser.add_argument('--CR', type=float, default=5, help='class ratios of minority')
    parser.add_argument('--num_feature', type=int, default=5, help='original feature length')

    
    args = parser.parse_args()

    return args