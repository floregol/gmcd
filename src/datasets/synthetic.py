import numpy as np
import torch.utils.data as data
import os
import pickle as pk
import math
"""
Dataset class for creating the shuffling dataset. 
"""


class SyntheticDataset(data.Dataset):
    def __init__(self,
                 train=False,
                 val=False,
                 test=False,
                 S=None,
                 K=None,
                 dataset_name="",
                 num_resample=1):
        self.S = S
        self.K = K
        self.N = 10000  # size of the dataset
        self.dataset_name = dataset_name
        self.np_data = None
        self.num_resample = num_resample
        self.pmax_vs_pmin = (2**(self.num_resample) - 1)
        U_pos = np.prod([self.K - i for i in range(self.S)])
        
        self.p_likely =1 / (U_pos / 2 + U_pos / (2 * self.pmax_vs_pmin))
        self.p_rare = self.p_likely / self.pmax_vs_pmin
        should_be_one = self.p_rare * U_pos / 2 + self.p_likely * U_pos / 2
        print('p rare', self.p_rare * U_pos/2, ' p likely', self.p_likely * U_pos / 2)
        self.data_path = 'experiments/synthetic/datasets/'
        self.data_path = os.path.join(self.data_path,
                                      dataset_name + str(self.pmax_vs_pmin))
        if self.S == self.K:
            dataset_file_name = str(self.S)
        else:
            dataset_file_name = str(self.S) + '_K_' + str(self.K)
        self.data_path = os.path.join(self.data_path, dataset_file_name)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.name = 'train'
        if val or test:
            self.name = 'val' if val else 'test'
        file_name = self.name + '.pk'
        filedict_name = self.name + '_dict.pk'
        file_path = os.path.join(self.data_path, file_name)
        file_dict = os.path.join(self.data_path, filedict_name)
        if not os.path.exists(file_path):
            print(file_path)
            print('dataset not existing, generating it')
            np.random.seed(3)  # train seed

            if val or test:
                np.random.seed(6 if val else 5)  # val or test seed

            self.np_data = np.stack(
                [self._generate_shuffle() for _ in range(self.N)])

            self.dict_permu = self.compute_all_example_dataset()
            with open(file_path, 'wb') as f:
                pk.dump(self.np_data, f)
            with open(file_dict, 'wb') as f:
                pk.dump(self.dict_permu, f)

        else:
            with open(file_path, 'rb') as f:
                self.np_data = pk.load(f)
            with open(file_dict, 'rb') as f:
                self.dict_permu = pk.load(f)

        size_support = np.prod([self.K - i for i in range(self.S)])
        num_examples = len(self.dict_permu.keys())
        print(self.name, 'covers', 100 * num_examples / size_support,
              '% of the support')

    def __len__(self):
        return self.np_data.shape[0]

    def __getitem__(self, idx):
        if self.np_data is None:
            return self._generate_shuffle()
        else:
            return self.np_data[idx]

    def test_likely(self, sample):
        if self.dataset_name == 'pair':
            exit()
        elif self.dataset_name == 'sort':
            return sample[0] < sample[-1]

    def _generate_shuffle(self):
        if self.dataset_name == 'pair':
            count = 0
            while count < self.num_resample:
                sample = np.random.permutation(self.K)[:self.S]
                if self.test_likely(sample):
                    return sample
                count += 1
        elif self.dataset_name == 'sort':
            count = 0
            while count < self.num_resample:
                sample = np.random.permutation(self.K)[:self.S]
                if self.test_likely(sample):
                    return sample
                count += 1
            return sample

    def test_if_valid_perm(self, list_x):
        dict_category = {}
        for x in list_x:
            if x in dict_category:
                return False
            elif x > self.K:
                return False
            else:
                dict_category[x] = 1
        return True

    def map_sequence_to_p(self, list_x):
        
        if self.test_if_valid_perm(list_x):
            if self.test_likely(list_x):
                return self.p_likely
            else:
                return self.p_rare

        else:
            return 0

    def get_all_p(self):
        
        return {0: {}, self.p_likely: {}, self.p_rare: {}}

    def samples_to_dict(self, samples_x):
        histogram_samples_per_p = self.get_all_p()

        for x in samples_x:
            list_x = list(x)
            string_key = "-".join(map(str, list_x))
            p = self.map_sequence_to_p(list_x)
            if string_key in histogram_samples_per_p[p]:
                histogram_samples_per_p[p][string_key] += 1
            else:
                histogram_samples_per_p[p][string_key] = 1
        return histogram_samples_per_p

    def get_size_support_dict(self):
        size_pos_support = np.prod([self.K - i for i in range(self.S)])
        size_support = self.K**self.S
       
        return {
            self.p_likely: size_pos_support / 2,
            self.p_rare: size_pos_support / 2,
            0: size_support - size_pos_support
        }

    def compute_all_example_dataset(self):
        dict_permu = {}
        for i in range(self.np_data.shape[0]):
            x = self.np_data[i, :]
            string_x = "-".join(map(str, x))
            if string_x in dict_permu:
                dict_permu[string_x] += 1
            else:
                dict_permu[string_x] = 1
        return dict_permu
