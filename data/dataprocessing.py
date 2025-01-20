import numpy as np
import random
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
import os
import shutil
import json

import torch
from torch.utils.data import Dataset

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def data_description(data_name):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = rf'{curr_dir}/datasets/{data_name}/{data_name}.xlsx'
    if os.path.exists(datapath):
        data = pd.read_excel(datapath).values.astype(float)
    else:
        datapath = rf'{curr_dir}/datasets/{data_name}/{data_name}.csv'
        data = pd.read_csv(datapath).values.astype(float)


    values_min = data.min(axis=0).astype(float)
    values_max = data.max(axis=0).astype(float)
    json_SOS = []
    for i in range(data.shape[1]):
        counts = Counter(data[:, i])
        counts_value = list(counts.keys())
        counts_value.sort()
        counts_len = counts.__len__()
        ideal_len = max(counts_value) - min(counts_value) + 1

        if (counts_len >= 0.9 * ideal_len) & (counts_len <= ideal_len):
            if (counts_len < 10) & (min(counts_value) >= 0):
                # categories
                if i == data.shape[1]-1:
                    json_SOS.append(
                        {
                            "i2s": counts_value,
                            "name": f"label",
                            "size": counts_len,
                            "type": "categorical"
                        }
                    )
                else:
                    json_SOS.append(
                        {
                            "i2s": counts_value,
                            "name": f"X{i}",
                            "size": counts_len,
                            "type": "categorical"
                        }
                    )
            else:
                # ordinal
                json_SOS.append(
                    {
                        "i2s": counts_value,
                        "name": f"X{i}",
                        "size": counts_len,
                        "type": "ordinal"
                    }
                )
        else:
            json_SOS.append(
                {
                    "max": values_max[i],
                    "min": values_min[i],
                    "name": f"X{i}",
                    "type": "continuous"
                }
            )

    json_SOS_final = {"columns": json_SOS, "problem_type": "binary_classification"}
    with open(rf'{curr_dir}/datasets/{data_name}/{data_name}.json', 'w') as f:
        f.write(json.dumps(json_SOS_final, ensure_ascii=False, indent=4, separators=(',', ':'), default=default_dump))

    for i in range(5):
        shutil.copy(rf'{curr_dir}/datasets/{data_name}/{data_name}.json',
                    rf'{curr_dir}/datasets/{data_name}/SOS/exp{i}/{data_name}.json')


def data_GOIO(data_name):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    for exp in range(5):

        data = np.load(rf'{curr_dir}/datasets/{data_name}/SOS/exp{exp}/{data_name}.npz')
        with open(rf'{curr_dir}/datasets/{data_name}/{data_name}.json', 'r') as f:
            data_info = json.load(f)
        xtrain, ytrain = data['train'][:,:-1], data['train'][:,-1]
        xtest, ytest = data['test'][:,:-1], data['test'][:,-1]

        n_num_size = 0
        n_cat_size = 0
        num_col_idx = []
        cat_col_idx = []
        categories = []

        for i, ds in enumerate(data_info['columns']):
            if (ds['type'] == 'categorical') & (ds['name'] != 'label'):
                cat_col_idx.append(i)
                n_cat_size+=1
                categories.append(ds['size'])
            elif ds['type'] == 'continuous' or ds['type'] == 'ordinal':
                num_col_idx.append(i)
                n_num_size+=1


        X_cat_test= xtest[:,cat_col_idx]
        X_cat_train = xtrain[:,cat_col_idx]
        X_num_test = xtest[:,num_col_idx]
        X_num_train = xtrain[:,num_col_idx]


        savepath = rf'{curr_dir}/datasets/{data_name}/GOIO/exp{exp}'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        np.save(f'{savepath}/X_num_train',X_num_train)
        np.save(f'{savepath}/X_num_test',X_num_test)
        np.save(f'{savepath}/X_cat_train',X_cat_train)
        np.save(f'{savepath}/X_cat_test',X_cat_test)
        np.save(f'{savepath}/y_train',ytrain)
        np.save(f'{savepath}/y_test',ytest)
        np.save(f'{savepath}/x_train',xtrain)

        data_basic = {
            "name": data_name,
            "id": f'exp{exp}',
            "task_type": "binclass",
            "n_num_features": n_num_size,
            "n_cat_features": n_cat_size,
            "test_size": len(ytest),
            "train_size": len(ytrain),
                "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "categories":categories
        }

        with open(f'{savepath}/info.json', 'w') as f:
            f.write(
                json.dumps(data_basic, ensure_ascii=False, indent=4, separators=(',', ':'), default=default_dump))



def make_imbalance(data, rate, taskclass):
    idx = []
    categories = []

    j = 0
    rk = 0
    for i in taskclass:
        index_list = [a for a, b in enumerate(data[:, -1]) if b == i]
        lens = int(len(index_list) * rate[rk])

        idx = idx + random.sample(index_list, lens)

        categories = categories + [j] * lens
        rk += 1
        j = j + 1

    return data[idx,:-1], categories

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def Default_processing(args):
    data_name = args.dataname
    # import and normalize data
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = f'{curr_dir}/datasets/{data_name}/{data_name}.xlsx'
    if os.path.exists(datapath):
        data = pd.read_excel(datapath).values.astype(float)
    else:
        datapath = f'{curr_dir}/datasets/{data_name}/{data_name}.csv'
        data = pd.read_csv(datapath).values.astype(float)

    savepath = f'{curr_dir}/datasets/{data_name}'
    x = data[:,:-1]
    y = data[:,-1]

    # make path
    dir_num = ['CLASSIC', 'SOS', 'TEST']
    path_dir = []
    for n, dirs in enumerate(dir_num):
        path_dir.append(os.path.join(savepath, dirs))
        make_dir(path_dir[n])

    # 5-fold cross validation data partitioning
    skfolds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    dmax = np.max(x, axis=0)
    dmin = np.min(x, axis=0)
    x_norm = ((x - dmin) / (dmax - dmin + 1e-8)) * 2 - 1

    k = 0
    for train_idx, test_idx in skfolds.split(x, y):
        path_dir_exp = []
        for n, dirs in enumerate(path_dir):
            path_dir_exp.append(os.path.join(dirs, f'exp{k}'))
            make_dir(path_dir_exp[n])

        X_train_folds = x[train_idx]
        y_train_folds = y[train_idx]
        X_test_fold = x[test_idx]
        y_test_fold = y[test_idx]

        # normalization by the maximum and minimum value of the training set
        X_norm_train_folds = x_norm[train_idx]
        X_norm_test_folds = x_norm[test_idx]

        # CLASSIC
        np.save(os.path.join(path_dir_exp[0], 'xtrain.npy'), X_train_folds)
        np.save(os.path.join(path_dir_exp[0], 'ytrain.npy'), y_train_folds)

        # SOS & GAMO
        np.savez(os.path.join(path_dir_exp[1], f'{data_name}.npz'), train=data[train_idx], test=data[test_idx])

        # TEST
        np.save(os.path.join(path_dir_exp[2], 'xtest.npy'), X_norm_test_folds[:, :])
        np.save(os.path.join(path_dir_exp[2], 'ytest.npy'), y_test_fold)

        k=k+1

    data_description(data_name)

    data_GOIO(data_name)

def generate_data(means=10, num_attributes=5, num_instances=4000, class_ratios=35):


    # Step 1: 设置正负类的均值
    positive_mean = np.ones(num_attributes) * (means * 1)  # positive class mean shifts by 1 sd each domain
    negative_mean = np.ones(num_attributes) * 0  # negative class mean remains 0

    # Step 2: 生成正负类数据
    positive_data = np.random.normal(positive_mean, 1, size=(num_instances, num_attributes))
    negative_data = np.random.normal(negative_mean, 1, size=(num_instances, num_attributes))

     # Step 3: 为每个样本生成非线性特征和噪声
    def augment_features(data):
        # 原始数据添加噪声
        noise = np.random.uniform(0, 0.1, size=data.shape)
        data = data + noise

        # 非线性变换 (例如平方和三次幂)
        nonlinear = np.hstack([data**2, data**3])

        # 交互特征（特征之间的乘积）
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interaction = poly.fit_transform(data)


        # 拼接所有特征
        return np.hstack([nonlinear, interaction])

    # Step 3: 生成不同类别的实例比例

    # Calculate the number of instances in each class based on the ratio
    positive_count = int(num_instances * class_ratios / 100)
    negative_count = num_instances - positive_count

    positive_data = augment_features(positive_data)
    negative_data = augment_features(negative_data)

    # Sample the data for this ratio
    positive_samples = positive_data[:positive_count]
    negative_samples = negative_data[:negative_count]

    # Concatenate the two classes together to form the dataset
    dataset = np.vstack([positive_samples, negative_samples])
    labels = np.array([1] * positive_count + [0] * negative_count)

    # Shuffle the dataset
    permuted_indices = np.random.permutation(num_instances)
    dataset = dataset[permuted_indices]
    dataset = dataset + 0.08*np.random.rand(dataset.shape[0],dataset.shape[1])

    labels = labels[permuted_indices]

    # Create a DataFrame to store the dataset with labels
    dataset_df = pd.DataFrame(dataset, columns=[f"Attribute_{i+1}" for i in range(dataset.shape[1])])
    dataset_df['Label'] = labels

    return dataset_df

def data_syn(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = f'human_data_{args.means}_{args.CR}'
    filepath = f'{curr_dir}/datasets/{dataname}'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    datasets = generate_data(num_attributes=args.num_feature, means=args.means, class_ratios=args.CR)
    datasets.to_csv(f'{filepath}/{dataname}.csv', index=None)

    args.dataname = dataname

    Default_processing(args)
    data_description(dataname)
    data_GOIO(dataname)




class datasets(Dataset):
    def __init__(self, path):
        # import training data
        x = np.load(os.path.join(path,'xtrain.npy'))
        y = np.load(os.path.join(path,'ytrain.npy'))

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        data = self.x[index, :, :]
        label = self.y[index]

        return data, label

