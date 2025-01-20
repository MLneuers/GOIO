import torch
from utils import execute_function, get_args
import numpy as np
import os

seed_value = 2024
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/exp{args.exp}/{args.method}'
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)