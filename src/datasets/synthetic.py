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
                 num_resample=1,
                 dont_generate=False):
        self.S = S
        self.K = K
        self.N = 10000  # size of the dataset
        self.dataset_name = dataset_name
        self.np_data = None
        self.num_resample = num_resample
        self.pmax_vs_pmin = (2**(self.num_resample + 1) - 1)
        if self.dataset_name == 'pair':
            self.U_pos = self.K * int(self.K / 2)**(self.S - 1)
        elif self.dataset_name == 'sort':
            self.U_pos = np.prod([self.K - i for i in range(self.S)])

        self.p_likely = 1 / (self.U_pos / 2 + self.U_pos /
                             (2 * self.pmax_vs_pmin))
        self.p_rare = self.p_likely / self.pmax_vs_pmin
        should_be_one = self.p_rare * self.U_pos / 2 + self.p_likely * self.U_pos / 2
        print('p rare', self.p_rare * self.U_pos / 2, ' p likely',
              self.p_likely * self.U_pos / 2)
        if not dont_generate:
            self.data_path = 'experiments/synthetic/datasets/'
            self.data_path = os.path.join(
                self.data_path, dataset_name + str(self.pmax_vs_pmin))
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

                self.dict_dataset = self.compute_all_example_dataset()
                with open(file_path, 'wb') as f:
                    pk.dump(self.np_data, f)
                with open(file_dict, 'wb') as f:
                    pk.dump(self.dict_dataset, f)

            else:
                with open(file_path, 'rb') as f:
                    self.np_data = pk.load(f)
                with open(file_dict, 'rb') as f:
                    self.dict_dataset = pk.load(f)

            this_should_match = self.samples_to_dict(self.np_data)
            p_emp_likely = np.sum([
                val for _, val in this_should_match[self.p_likely].items()
            ]) / self.np_data.shape[0]
            p_emp_rare = np.sum([
                val for _, val in this_should_match[self.p_rare].items()
            ]) / self.np_data.shape[0]
            p_emp_0 = np.sum([val for _, val in this_should_match[0].items()
                              ]) / self.np_data.shape[0]
            num_examples = len(self.dict_dataset.keys())
            print(self.name, 'covers', 100 * num_examples / self.U_pos,
                  '% of the support')
            print('p_emp_likely', p_emp_likely, 'p_emp_rare', p_emp_rare,
                  'p_emp_0', p_emp_0)

    def __len__(self):
        return self.np_data.shape[0]

    def __getitem__(self, idx):
        if self.np_data is None:
            return self._generate_shuffle()
        else:
            return self.np_data[idx]

    def test_likely(self, sample):
        if self.dataset_name == 'pair':
            return (sample[0] + sample[-1]) % 2==0
        elif self.dataset_name == 'sort':
            return sample[0] < sample[-1]

    def get_pos_sample(self):
        if self.dataset_name == 'pair':
            sample = [np.random.choice([i for i in range(self.K)])]
            for i in range(self.S - 1):
                choices = [
                    i % self.K
                    for i in range(sample[i], sample[i] + int(self.K / 2))
                ]
                sample.append(np.random.choice(choices))
            return sample
        elif self.dataset_name == 'sort':
            sample = np.random.permutation(self.K)[:self.S]
            return sample

    def _generate_shuffle(self):

        count = 0
        sample = self.get_pos_sample()
        while count < self.num_resample:
            if self.test_likely(sample):
                return sample
            count += 1
            sample = self.get_pos_sample()
        return sample

    def test_if_valid_x(self, sample):
        if self.dataset_name == 'pair':

            if sample[0] not in [i for i in range(self.K)]:
                return False
            for i, x in enumerate(sample[1:]):
                choices = [
                    i % self.K
                    for i in range(sample[i], sample[i] + int(self.K / 2))
                ]
                if x not in choices:
                    return False
            return True
        elif self.dataset_name == 'sort':
            dict_category = {}
            for x in sample:
                if x in dict_category:
                    return False
                elif x > self.K:
                    return False
                else:
                    dict_category[x] = 1
            return True

    def map_sequence_to_p(self, list_x):

        if self.test_if_valid_x(list_x):
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
        dict_dataset = {}
        for i in range(self.np_data.shape[0]):
            x = self.np_data[i, :]
            string_x = "-".join(map(str, x))
            if string_x in dict_dataset:
                dict_dataset[string_x] += 1
            else:
                dict_dataset[string_x] = 1
        return dict_dataset
