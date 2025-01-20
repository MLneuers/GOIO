import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from CLDM.model import MLPDiffusion, Model
from CLDM.latent_utils import get_input_train
from utils_train import DiffDataset

warnings.filterwarnings('ignore')

def main(args): 
    device = args.device

    train_z, _, _, ckpt_path, _, proto, labels = get_input_train(args)


    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2



    ds = DiffDataset(train_z, labels)
    proto = proto.to(device)
    class_0 = torch.zeros_like(proto)
    proto = torch.concatenate([class_0[0][None,:],proto],axis=0)


    batch_size = 4096
    train_loader = DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True,
    )

    num_epochs = 10000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch[0].float().to(device)
            if args.condition:
                y = batch[1].long().to(device)
                idx = torch.where(torch.randn(y.shape[0]) < args.threshold)[0]
                y[idx.to(device)] = 0
                y = proto[y.to(torch.long)].to(device)
                y = y.to(torch.float)
            else:
                y =None
            loss, D_x, weights = model(inputs, class_labels=y)


            loss_all = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss_all.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 100:
                print('Early stopping')
                break


    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'