"""
This file has been modified from a file released under the Apache License 2.0.
Based on https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/main.py
"""

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import warnings
from einops import rearrange

import os
from tqdm import tqdm
import json
import time

from MLVAE.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset_withy

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def compute_loss2(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z, prototype, labels):
	ce_loss_fn = nn.CrossEntropyLoss()
	mse_loss = (X_num - Recon_X_num).pow(2).mean()
	ce_loss = 0
	acc = 0
	total_num = 0

	for idx, x_cat in enumerate(Recon_X_cat):
		if x_cat is not None:
			ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
			x_hat = x_cat.argmax(dim = -1)
		acc += (x_hat == X_cat[:,idx]).float().sum()
		total_num += x_hat.shape[0]

	ce_loss /= (idx + 1)
	acc /= total_num

	ce_loss_fn = nn.CrossEntropyLoss()

	mu_z_dist = rearrange(mu_z, 'a b c->a (b c)')
	prototype_dist = rearrange(prototype, 'a b c->a (b c)')


	dist = torch.zeros(mu_z.shape[0], prototype.shape[0], device='cuda:0')
	new_mu_z = torch.zeros_like(mu_z)
	for i in range(prototype.shape[0]):
		temp = torch.cosine_similarity(mu_z_dist.float().detach(), prototype_dist[i].float(), axis=1)
		dist[:, i] = temp

	temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()



	proto_ce_loss = ce_loss_fn(dist, labels.squeeze())
	proto_ce_loss = proto_ce_loss.requires_grad_(True)

	loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

	acc_label = ((dist.argmax(dim=-1) == labels.squeeze()).sum())/labels.shape[0]

	return mse_loss, ce_loss, loss_kld, acc, proto_ce_loss, acc_label



def main(args):
	dataname = args.dataname


	max_beta = args.max_beta
	min_beta = 0.005
	lambd = args.lambd

	device =  args.device


	data_dir = rf'./data/datasets/{dataname}/GOIO/exp{args.exp}'
	info_path = rf'./data/datasets/{dataname}/GOIO/exp{args.exp}/info.json'

	with open(info_path, 'r') as f:
		info = json.load(f)

	ckpt_dir = f'./ckpt_{args.kld}_{args.proto}_{args.dist}//{dataname}/exp{args.exp}'
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	model_save_path = f'{ckpt_dir}/model.pt'
	encoder_save_path = f'{ckpt_dir}/encoder.pt'
	decoder_save_path = f'{ckpt_dir}/decoder.pt'

	X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])

	categories = [categories[0]]+info['categories']

	X_train_num, X_test_num = X_num
	X_train_cat, X_test_cat = X_cat

	X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
	X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

	y_train = np.load(f'{data_dir}/y_train.npy')
	y_test = np.load(f'{data_dir}/y_test.npy')

	class_num = np.unique(y_train).shape[0]

	y_train = torch.tensor(y_train, device='cuda:0').long()
	y_test = torch.tensor(y_test, device='cuda:0').long()


	train_data = TabularDataset_withy(X_train_num.float(), X_train_cat, y_train)

	X_test_num = X_test_num.float().to(device)
	X_test_cat = X_test_cat.to(device)

	batch_size = 4096
	train_loader = DataLoader(
		train_data,
		batch_size = batch_size,
		shuffle = True,
	)

	model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True,prototype_shape=[class_num,d_numerical+len(categories),D_TOKEN])
	model = model.to(device)


	pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
	pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

	pre_encoder.eval()
	pre_decoder.eval()

	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

	num_epochs = 5000
	best_train_loss = float('inf')

	current_lr = optimizer.param_groups[0]['lr']
	patience = 0

	beta = max_beta
	start_time = time.time()
	prototype = None
	with tqdm(total=num_epochs) as pbar:
		for epoch in range(num_epochs):
			pbar.update(1)
			pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

			curr_loss_multi = 0.0
			curr_loss_gauss = 0.0
			curr_loss_kl = 0.0
			curr_loss_pt = 0.0
			curr_loss_dis = 0.0

			curr_count = 0

			for batch_num, batch_cat, batch_label in train_loader:
				model.train()
				optimizer.zero_grad()

				batch_num = batch_num.to(device)
				batch_cat = batch_cat.to(device)
				batch_label = batch_label.to(device)

				Recon_X_num, Recon_X_cat, mu_z, std_z, prototype, dist_loss = model(batch_num, batch_cat, batch_label)

				loss_mse, loss_ce, loss_kld, train_acc, loss_pt, acc_train = compute_loss2(batch_num, batch_cat, Recon_X_num,
																						   Recon_X_cat, mu_z[:,1:,:], std_z[:,1:,:],
																						   prototype,batch_label)

				loss = loss_mse + loss_ce + beta*(args.kld*loss_kld + args.dist*dist_loss ) + args.proto*loss_pt
				loss.backward()
				optimizer.step()


				batch_length = batch_num.shape[0]
				curr_count += batch_length
				curr_loss_multi += loss_ce.item() * batch_length
				curr_loss_gauss += loss_mse.item() * batch_length
				curr_loss_kl    += loss_kld.item() * batch_length
				curr_loss_pt += loss_pt.item() * batch_length
				curr_loss_dis += dist_loss.item() * batch_length

			num_loss = curr_loss_gauss / curr_count
			cat_loss = curr_loss_multi / curr_count
			kl_loss = curr_loss_kl / curr_count
			pt_loss = curr_loss_pt / curr_count
			dist_loss = curr_loss_dis / curr_count

			'''
				Evaluation
			'''
			model.eval()
			with torch.no_grad():
				Recon_X_num, Recon_X_cat, mu_z, std_z,_,val_dist = model(X_test_num, X_test_cat ,y_test)

				val_mse_loss, val_ce_loss, val_kl_loss, val_acc, val_loss_pt, acc_val = compute_loss2(X_test_num,
																									  X_test_cat,
																									  Recon_X_num,
																									  Recon_X_cat,
																									  mu_z[:, 1:, :],
																									  std_z[:, 1:, :],
																									  prototype,
																									  y_test)
				val_loss = val_mse_loss.item() + val_ce_loss.item()

				scheduler.step(val_loss)
				new_lr = optimizer.param_groups[0]['lr']

				if new_lr != current_lr:
					current_lr = new_lr
					print(f"Learning rate updated: {current_lr}")

				train_loss = val_loss
				if train_loss < best_train_loss:
					best_train_loss = train_loss
					patience = 0
					torch.save(model.state_dict(), model_save_path)
				else:
					patience += 1
					if patience == 10:
						if beta > min_beta:
							beta = beta * lambd

			print(
				'epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train dsit:{:.6f},'
				' Val MSE:{:.6f}, Val CE:{:.6f}, Train pt:{:6f}, Val pt:{:6f} Val dist:{:6f}, acc_t:{:4f} acc_v:{:4f}'.format(
					epoch, beta, num_loss, cat_loss, kl_loss, dist_loss, val_mse_loss.item(), val_ce_loss.item(),
					pt_loss,
					val_loss_pt.item(), val_dist.item(), acc_train.item(), acc_val.item()))


	end_time = time.time()
	print('Training time: {:.4f} mins'.format((end_time - start_time)/60))

	# Saving latent embeddings
	with torch.no_grad():
		pre_encoder.load_weights(model)
		pre_decoder.load_weights(model)

		torch.save(pre_encoder.state_dict(), encoder_save_path)
		torch.save(pre_decoder.state_dict(), decoder_save_path)

		X_train_num = X_train_num.to(device)
		X_train_cat = X_train_cat.to(device)

		print('Successfully load and save the model!')

		train_z, log_var = pre_encoder(X_train_num, X_train_cat)

		np.save(f'{ckpt_dir}/train_z.npy', train_z.detach().cpu().numpy())
		np.save(f'{ckpt_dir}/log_var.npy', log_var.detach().cpu().numpy())

		prototype = model.proto.detach().cpu().numpy()
		np.save(f'{ckpt_dir}/prototype.npy', prototype)


		print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Variational Autoencoder')

	parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
	parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
	parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
	parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
	parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

	args = parser.parse_args()

	# check cuda
	if args.gpu != -1 and torch.cuda.is_available():
		args.device = 'cuda:{}'.format(args.gpu)
	else:
		args.device = 'cpu'