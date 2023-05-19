import numpy as np
import torch.utils.data as data
import os
import pickle as pk
from scipy.stats import rv_discrete
"""
Dataset class for protein dataset. 
"""

NUM_ACIDO = 21


class RealDataset(data.Dataset):
    def __init__(self,
                 train=False,
                 val=False,
                 test=False,
                 dataset_name="",
                 dont_generate=False):
        self.dataset_name = dataset_name
        name = 'PF00014'
        self.data_path = 'src/datasets'
        if dataset_name == 'proxy':
            self.file_name = name + '_proxy.pk'
            self.pmf = self.load_pk_file()
            p = list(self.pmf.values())
            x = list(self.pmf.keys())
            indices = list(range(len(p)))
            index_samples = self.sample_pmf(indices, p, m=10000)
            samples = [np.array(x[i]) for i in index_samples]
            self.np_data = np.array(samples)
        else:
            self.data_path = 'src/datasets'
            self.file_name = name + '_mgap6.npy'

            self.np_data = self.load_data()

        self.num_classes = NUM_ACIDO
        self.set_size = self.np_data.shape[1]
        self.dataset_size = self.np_data.shape[0]
        if not dont_generate:
            self.train_percent = 0.7
            self.val_percent = 0.1
            self.test_percent = 0.2
            int_train = int(self.dataset_size * self.train_percent)
            int_val = int_train + int(self.dataset_size * self.val_percent)
            if train:
                self.index = list(range(0, int_train))
            if val:
                self.index = list(range(int_train, int_val))
            if test:
                self.index = list(range(int_val, self.dataset_size))
            self.np_data = self.np_data[self.index, :]

    def sample_pmf(self, value, probability, m):
        distrib = rv_discrete(values=(value, probability))
        new_samples = distrib.rvs(size=m)
        return new_samples

    def get_vocab_size():
        return NUM_ACIDO

    def get_set_size(name):
        if name == 'PF00014':
            return 53
        elif name == 'PF00076':
            return 70
        else:
            return None

    def load_data(self):
        filepath = os.path.join(self.data_path, self.file_name)
        return np.load(filepath)

    def load_pk_file(self):
        filepath = os.path.join(self.data_path, self.file_name)
        with open(filepath, 'rb') as f:
            data = pk.load(f)
        return data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.np_data[idx, :]
