from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import warnings


_MODELS = {
	'binary_classification': [
		{
			'class': GaussianNB,
			'kwargs': {

			}
		},
		{
			'class': DecisionTreeClassifier,
			'kwargs': {
				'criterion':'gini'
			}
		},

		{
			'class': SVC,
			'kwargs': {
			}
		},

		{
			'class': RandomForestClassifier,
			'kwargs': {

			}
		},


		{
			'class': KNeighborsClassifier,
			'kwargs': {
				'n_neighbors': 5
			}
		},

	]
}
warnings.simplefilter(action='ignore', category=FutureWarning)


def indicator_cls(labels, predicted):
	accuracy = accuracy_score(labels, predicted)
	macro_f1 = metrics.f1_score(labels, predicted, average='macro')
	mcc = metrics.matthews_corrcoef(labels, predicted)

	return {"accuracy": accuracy,
			"macro_f1": macro_f1,
			"mcc":mcc,
}




def main(args):
	dataname = args.dataname
	device = args.device
	save_path = args.save_path

	dataset_path = rf'./data/datasets/{dataname}'
	raw_data = np.load(f'{dataset_path}/SOS/exp{args.exp}/{dataname}.npz')
	syn_data = pd.read_csv(f'{save_path}/syn_{dataname}.csv').values

	train_raw_X, train_raw_Y = raw_data['train'][:, :-1], raw_data['train'][:, -1]
	test_X, test_Y = raw_data['test'][:, :-1], raw_data['test'][:, -1]
	vmax, vmin = train_raw_X.max(0), train_raw_X.min(0)
	norm_test_X = 2 * (test_X - vmin) / (vmax - vmin+1e-8) - 1


	major = syn_data[np.where(syn_data[:, -1] == 0), :].squeeze()
	minor = syn_data[np.where(syn_data[:, -1] == 1), :].squeeze()
	raw_major = train_raw_X[np.where(train_raw_Y[:] == 0), :].squeeze()
	raw_minor = train_raw_X[np.where(train_raw_Y[:] == 1), :].squeeze()
	final_num = 1.0 * raw_major.shape[0]

	fixed_result = []
	classifiers = _MODELS['binary_classification']
	for i, classifier in enumerate(classifiers):

		for k in range(10):
			idx_maj = np.random.choice(major.shape[0], int(final_num - raw_major.shape[0]), replace=False)
			try:
				idx_min = np.random.choice(minor.shape[0], int(final_num - raw_minor.shape[0]), replace=False)
			except ValueError:
				idx_min = np.random.choice(minor.shape[0], int(final_num - raw_minor.shape[0]), replace=True)

			syn_X_maj, syn_y_maj = major[idx_maj, :-1], major[idx_maj, -1]
			syn_X_min, syn_y_min = minor[idx_min, :-1], minor[idx_min, -1]
			fixed_data_X, fixed_data_Y = np.vstack([raw_data['train'][:, :-1], syn_X_maj, syn_X_min]), np.hstack(
				[raw_data['train'][:, -1], syn_y_maj, syn_y_min])

			norm_fixed_data_X = 2 * (fixed_data_X - vmin) / (vmax - vmin+1e-8) - 1

			model_param = classifier['kwargs']
			model = classifier['class'](**model_param)
			model.fit(norm_fixed_data_X, fixed_data_Y)
			pred = model.predict(norm_test_X)
			result = indicator_cls(test_Y, pred)
			fixed_result.append(result)
	fixed_result = pd.DataFrame(fixed_result)
	means = fixed_result.mean(axis=0)
	std = fixed_result.std(axis=0)
	fixed_result = fixed_result._append(means, ignore_index=True)
	fixed_result = fixed_result._append(std, ignore_index=True)

	fixed_result.to_csv(f'{save_path}/{dataname}.csv')
