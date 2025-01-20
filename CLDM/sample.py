import numpy as np
import torch

import argparse
import warnings
import time
import os

from CLDM.model import MLPDiffusion, Model
from CLDM.latent_utils import get_input_generate
from CLDM.diffusion_utils import sample
from collections import Counter
from utils_train import  split_num_cat_target, recover_data
import pandas as pd

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    save_path = args.save_path

    train_z, prototype, train_y, x, ckpt_path, info, num_inverse, cat_inverse, type_order = get_input_generate(args)

    class_0 = torch.zeros_like(prototype)
    prototype = torch.concatenate([class_0[0][None,:],prototype],axis=0)
    in_dim = train_z.shape[1]
    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0] * 10
    sample_dim = in_dim

    oversample_rate = 1.1
    init_num = [j for _, j in sorted(Counter(train_y.squeeze()).items())]  # The number of each category
    final_num = int(max(init_num) * oversample_rate)
    syn_data_y = []
    syn_data = None


    if args.condition:

        for i, j in enumerate(init_num):
            sample_batch = int(final_num - j)
            cl = [i] * sample_batch * 5+[0] * sample_batch * 5
            classidx =  [i+1 for k in range(sample_batch * 5)]+[0 for k in range(sample_batch * 5)]
            condition = torch.tensor(prototype[classidx]).to(device)

            x_next = sample(model.denoise_fn_D, len(classidx), sample_dim, class_labels=condition)
            x_next = x_next * 2 + mean.to(device)
            x_next = x_next[:int(sample_batch*5)]

            if syn_data is None:
                syn_data = x_next.float().cpu().numpy()
            else:
                syn_data = np.vstack([syn_data,x_next.float().cpu().numpy()])

            syn_data_y += cl[:int(sample_batch*5)]

    else:
        x_next = sample(model.denoise_fn_D, num_samples, sample_dim, class_labels=None)
        x_next = x_next * 2 + mean.to(device)
        syn_data = x_next.float().cpu().numpy()


    syn_num, syn_cat = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device)
    syn_df = recover_data(syn_num, syn_cat, type_order)
    syn_df = pd.DataFrame(syn_df)


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    syn_df.to_csv(f'{save_path}/syn_{dataname}.csv', index=False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='abalone', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'