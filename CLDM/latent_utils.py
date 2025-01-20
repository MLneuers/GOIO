"""
This file has been modified from a file released under the Apache License 2.0.
Based on https://github.com/amazon-science/tabsyn/blob/main/tabsyn/latent_utils.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from utils_train import preprocess
from MLVAE.model import Decoder_model


def get_input_train(args):
	dataname = args.dataname

	MLVAE_path = f'./ckpt_{args.kld}_{args.proto}_{args.dist}/{dataname}/exp{args.exp}'
	CLDM_path = f'./ckpt_{args.kld}_{args.proto}_{args.dist}/{dataname}/exp{args.exp}/'
	dataset_dir = rf'./data/datasets/{dataname}/GOIO/exp{args.exp}'

	with open(f'{dataset_dir}/info.json', 'r') as f:
		info = json.load(f)

	embedding_save_path = f'{MLVAE_path}/train_z.npy'
	proto_save_path = f'{MLVAE_path}/prototype.npy'
	labels_save_path = f'{dataset_dir}/y_train.npy'

	train_z = torch.tensor(np.load(embedding_save_path)).float()
	proto = torch.tensor(np.load(proto_save_path)).float()
	labels = torch.tensor(np.load(labels_save_path)).long()

	train_z = train_z[:, 1:, :]
	B, num_tokens, token_dim = train_z.size()
	in_dim = num_tokens * token_dim

	train_z = train_z.view(B, in_dim)
	proto = proto.view(proto.shape[0], in_dim)

	return train_z, MLVAE_path, dataset_dir, CLDM_path, info, proto, labels


def get_input_generate(args):
	dataname = args.dataname

	MLVAE_path = f'./ckpt_{args.kld}_{args.proto}_{args.dist}/{dataname}/exp{args.exp}'
	CLDM_path = f'./ckpt_{args.kld}_{args.proto}_{args.dist}/{dataname}/exp{args.exp}'
	dataset_dir = rf'./data/datasets/{dataname}/GOIO/exp{args.exp}'
	datadir = f'./data/datasets/{dataname}'

	with open(f'{datadir}/{dataname}.json', 'r') as f:
		datainfo = json.load(f)
	datainfo = datainfo['columns']
	type_order = []
	for i in range(len(datainfo)):
		if datainfo[i]['name'] != 'label':
			type_order.append(datainfo[i]['type'])

	with open(f'{dataset_dir}/info.json', 'r') as f:
		info = json.load(f)

	task_type = info['task_type']

	_, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(dataset_dir, task_type=task_type, inverse=True)
	categories = [categories[0]] + info['categories']

	train_z = torch.tensor(np.load(f'{MLVAE_path}/train_z.npy')).float()
	prototype = torch.tensor(np.load(f'{MLVAE_path}/prototype.npy')).float()

	train_z = train_z[:, 1:, :]

	B, num_tokens, token_dim = train_z.size()
	in_dim = num_tokens * token_dim

	train_z = train_z.view(B, in_dim)
	prototype = prototype.view(-1, in_dim)
	pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)

	decoder_save_path = f'{MLVAE_path}/decoder.pt'
	pre_decoder.load_state_dict(torch.load(decoder_save_path))

	info['pre_decoder'] = pre_decoder
	info['token_dim'] = token_dim

	y = np.load(f'{dataset_dir}/y_train.npy')
	x = np.load(f'{dataset_dir}/x_train.npy')

	return train_z, prototype, y, x, CLDM_path, info, num_inverse, cat_inverse, type_order


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
	task_type = info['task_type']

	num_col_idx = info['num_col_idx']
	cat_col_idx = info['cat_col_idx']
	target_col_idx = info['target_col_idx']

	n_num_feat = len(num_col_idx)
	n_cat_feat = len(cat_col_idx)

	if task_type == 'regression':
		n_num_feat += len(target_col_idx)
	else:
		n_cat_feat += len(target_col_idx)

	pre_decoder = info['pre_decoder']
	token_dim = info['token_dim']

	syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

	norm_input = pre_decoder(torch.tensor(syn_data))
	x_hat_num, x_hat_cat = norm_input

	syn_cat = []
	for pred in x_hat_cat:
		syn_cat.append(pred.argmax(dim=-1))

	syn_num = x_hat_num.cpu().numpy()
	syn_cat = torch.stack(syn_cat).t().cpu().numpy()

	syn_num = num_inverse(syn_num)
	syn_cat = cat_inverse(syn_cat)

	if info['task_type'] == 'regression':
		syn_target = syn_num[:, :len(target_col_idx)]
		syn_num = syn_num[:, len(target_col_idx):]

	else:
		print(syn_cat.shape)
		syn_target = syn_cat[:, :len(target_col_idx)]
		syn_cat = syn_cat[:, len(target_col_idx):]

	return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):
	num_col_idx = info['num_col_idx']
	cat_col_idx = info['cat_col_idx']
	target_col_idx = info['target_col_idx']

	idx_mapping = info['idx_mapping']
	idx_mapping = {int(key): value for key, value in idx_mapping.items()}

	syn_df = pd.DataFrame()

	if info['task_type'] == 'regression':
		for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
			if i in set(num_col_idx):
				syn_df[i] = syn_num[:, idx_mapping[i]]
			elif i in set(cat_col_idx):
				syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
			else:
				syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


	else:
		for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
			if i in set(num_col_idx):
				syn_df[i] = syn_num[:, idx_mapping[i]]
			elif i in set(cat_col_idx):
				syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
			else:
				syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

	return syn_df


def process_invalid_id(syn_cat, min_cat, max_cat):
	syn_cat = np.clip(syn_cat, min_cat, max_cat)

	return syn_cat
