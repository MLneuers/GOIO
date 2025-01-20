import numpy as np
import os

import src
from torch.utils.data import Dataset
import torch
from collections import Counter

class TabularDataset_withy(Dataset):
	def __init__(self, X_num, X_cat, y):
		self.X_num = X_num
		self.X_cat = X_cat
		self.y = y

	def __getitem__(self, index):
		this_num = self.X_num[index]
		this_cat = self.X_cat[index]
		this_label = self.y[index]

		sample = (this_num, this_cat, this_label)

		return sample

	def __len__(self):
		return self.X_num.shape[0]

class DiffDataset(Dataset):
	def __init__(self, X, y):
		self.X = X[:,:]
		self.y = y.squeeze()

	def __getitem__(self, index):
		this_X = self.X[index]
		this_label = self.y[index]

		return this_X, this_label+1

	def __len__(self):
		return self.X.shape[0]

def recover_data(syn_num, syn_cat, type_order):

	syn_od = []
	num_idx = 0
	cat_idx = 0

	for order in type_order:
		if order == 'categorical':
			syn_od.append(syn_cat[:,cat_idx+1])
			cat_idx += 1
		elif order == 'continuous' or order == 'ordinal':
			syn_od.append(syn_num[:,num_idx])
			num_idx += 1
	# label
	syn_od.append(syn_cat[:, 0])

	# syn_df = pd.DataFrame(syn_od).T
	syn_data = np.array(syn_od).T
	syn_data = syn_data.astype(float)

	return syn_data

def get_input_train(args):
	dataname = args.dataset

	curr_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = rf'D:\pythonproject\second_article\data/{dataname}/TABM/{args.exp}'

	ckpt_dir = f'{curr_dir}/tabsyn/vae/ckpt_{args.kld}_{args.proto}_{args.dist}/{dataname}/'
	train_z = torch.tensor(np.load(f'{ckpt_dir}/train_z.npy')).float()
	prototype = torch.tensor(np.load(f'{ckpt_dir}/prototype.npy')).float()
	y_train = torch.tensor(np.load(f'{dataset_dir}/y_train.npy')).float()

	if args.compensate:
		log_var = torch.tensor(np.load(f'{ckpt_dir}/log_var.npy')).float()
		init_num = [j for _, j in sorted(Counter(np.array(y_train).squeeze()).items())]  # The number of each category
		final_num = int(max(init_num)*1.1)
		syn_data_y = []
		syn_data = None
		for i, j in enumerate(init_num):
			sample_batch = int(final_num - j)
			classidx = [i] * sample_batch

			index = torch.where(y_train == i)
			task_mu, task_var = train_z[index[0],:,:], log_var[index[0],:,:]
			idx = np.random.randint(0, task_mu.shape[0], sample_batch)
			std = torch.exp(0.5*task_var[idx,:,:])
			eps = torch.randn_like(std)
			newdata = task_mu[idx,:,:] + eps * std

			if syn_data is None:
				syn_data = newdata.clone()
			else:
				syn_data = torch.vstack([syn_data,newdata])

			syn_data_y += classidx
		syn_data_y = np.array(syn_data_y).astype(int)
		train_z = torch.vstack([train_z,syn_data])
		y_train = np.vstack([y_train,syn_data_y[:,None]])


	train_z = train_z[:, 1:, :]


	B, num_tokens, token_dim = train_z.size()
	in_dim = num_tokens * token_dim

	train_z = train_z.view(B,1,in_dim)
	prototype = prototype.view(-1,1,in_dim)

	auxiliary = prototype.view(-1,in_dim)



	return train_z, ckpt_dir, prototype, auxiliary, y_train

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
	num_col_idx = info['num_col_idx']
	cat_col_idx = info['cat_col_idx']


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


	return syn_num, syn_cat


class TabularDataset_withy(Dataset):
	def __init__(self, X_num, X_cat, y):
		self.X_num = X_num
		self.X_cat = X_cat
		self.y = y

	def __getitem__(self, index):
		this_num = self.X_num[index]
		this_cat = self.X_cat[index]
		this_label = self.y[index]

		sample = (this_num, this_cat, this_label)

		return sample

	def __len__(self):
		return self.X_num.shape[0]

class TabularDataset(Dataset):
	def __init__(self, X_num, X_cat):
		self.X_num = X_num
		self.X_cat = X_cat

	def __getitem__(self, index):
		this_num = self.X_num[index]
		this_cat = self.X_cat[index]

		sample = (this_num, this_cat)

		return sample

	def __len__(self):
		return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):

	T_dict = {}

	T_dict['normalization'] = "quantile"
	T_dict['num_nan_policy'] = 'mean'
	T_dict['cat_nan_policy'] =  None
	T_dict['cat_min_frequency'] = None
	T_dict['cat_encoding'] = cat_encoding
	T_dict['y_policy'] = "default"

	T = src.Transformations(**T_dict)

	dataset = make_dataset(
		data_path = dataset_path,
		T = T,
		task_type = task_type,
		change_val = False,
		concat = concat
	)

	if cat_encoding is None:
		X_num = dataset.X_num
		X_cat = dataset.X_cat

		X_train_num, X_test_num = X_num['train'], X_num['test']
		X_train_cat, X_test_cat = X_cat['train'], X_cat['test']

		categories = src.get_categories(X_train_cat)
		d_numerical = X_train_num.shape[1]

		X_num = (X_train_num, X_test_num)
		X_cat = (X_train_cat, X_test_cat)


		if inverse:
			num_inverse = dataset.num_transform.inverse_transform
			cat_inverse = dataset.cat_transform.inverse_transform

			return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
		else:
			return X_num, X_cat, categories, d_numerical
	else:
		return dataset


def update_ema(target_params, source_params, rate=0.999):
	"""
	Update target parameters to be closer to those of source parameters using
	an exponential moving average.
	:param target_params: the target parameter sequence.
	:param source_params: the source parameter sequence.
	:param rate: the EMA rate (closer to 1 means slower).
	"""
	for target, source in zip(target_params, source_params):
		target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
	if X is None:
		return y.reshape(-1, 1)
	return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
	data_path: str,
	T: src.Transformations,
	task_type,
	change_val: bool,
	concat = True, #是否把类别加入到特征的学习中
):

	# classification
	if task_type == 'binclass' or task_type == 'multiclass':
		X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
		X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
		y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

		for split in ['train', 'test']:
			X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
			if X_num is not None:
				X_num[split] = X_num_t
			if X_cat is not None:
				if concat:
					X_cat_t = concat_y_to_X(X_cat_t, y_t)
				X_cat[split] = X_cat_t
			if y is not None:
				y[split] = y_t
	else:
		# regression
		X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
		X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
		y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

		for split in ['train', 'test']:
			X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

			if X_num is not None:
				if concat:
					X_num_t = concat_y_to_X(X_num_t, y_t)
				X_num[split] = X_num_t
			if X_cat is not None:
				X_cat[split] = X_cat_t
			if y is not None:
				y[split] = y_t

	info = src.load_json(os.path.join(data_path, 'info.json'))

	D = src.Dataset(
		X_num,
		X_cat,
		y,
		y_info={},
		task_type=src.TaskType(info['task_type']),
		n_classes=info.get('n_classes')
	)

	if change_val:
		D = src.change_val(D)

	# def categorical_to_idx(feature):
	#     unique_categories = np.unique(feature)
	#     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
	#     idx_feature = np.array([idx_mapping[category] for category in feature])
	#     return idx_feature

	# for split in ['train', 'val', 'test']:
	# D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

	return src.transform_dataset(D, T, None)