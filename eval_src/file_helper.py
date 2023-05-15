import os
import pickle as pk
import math
from os import listdir
from os.path import isfile, join
from eval_src.sampling import generate_samples_scalable
from eval_src.sequence_converter import convert_key_sequence_to_int, sequence_to_id
from eval_src.stair_p import make_stair_prob
import numpy as np
from eval_src.stair_p import samples_to_histo
from src.datasets.synthetic import SyntheticDataset


def load_all_files(path):

    return [f for f in listdir(path) if isfile(join(path, f))]


def store_for_plotting(data, title):
    file_path = os.path.join('results', title + '.pk')
    with open(file_path, 'wb') as f:  # store the data
        pk.dump(data, f)


def read_pickle_file(filename, p):
    file_name = str(filename)
    path = str(p)  # argmaxAR case -- pretty graph
    # path = './S_6_K_6/CDM/07_08_2022__11_49/figure' # CDM option 1 -- not bad graph
    # later on should be over all .pk files
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as f:
        samples = pk.load(f)
    # print(samples)
    return samples


# this will try to load the data, if the data isn't there,
# it will call the generating_func which returns the data and store it.


def read_file_else_store(file_path, generating_func):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:  # load the data
            data = pk.load(f)
    else:
        print('Couldnt find', file_path, 'generating and storing...')
        data = generating_func()
        with open(file_path, 'wb') as f:  # store the data
            pk.dump(data, f)
    return data


# simple function to quickly generate name of files


def create_prefix_from_list(dict_name):
    list_word = []
    for key, val in dict_name.items():
        try:
            if isinstance(val, int):
                word = str(val)
            else:
                word = "{:.2f}".format(val)
        except Exception:
            word = val
        list_word.append(key + ':' + word)
    prefix_srt = '_'.join(list_word)
    return prefix_srt


# this function will either load existing samples or generate new ones


def load_samples(list_of_espilon_q, b, ground_truth_p, splits, U, m, S, ratio,
                 TYPE):
    # obtain the samples
    list_of_samples = []
    list_of_pmf_q = []
    directory_samples_file = 'samples_storing'
    for e in list_of_espilon_q:
        sample_file = create_prefix_from_list({
            'exp': 'SYNTH' + TYPE,
            'U': U,
            'm': m,
            'splits': splits,
            'S': S,
            'ratio': ratio,
            'b': b,
            'e': e
        }) + '_samples.pk'

        sample_file_path = os.path.join(directory_samples_file, sample_file)

        def generating_samples_func():
            if e == 0:
                samples_and_pmf = generate_samples_scalable(ground_truth_p,
                                                            splits,
                                                            U,
                                                            m,
                                                            tempered=False,
                                                            e=0,
                                                            b=100,
                                                            TYPE=TYPE)

            else:
                samples_and_pmf = generate_samples_scalable(ground_truth_p,
                                                            splits,
                                                            U,
                                                            m,
                                                            tempered=True,
                                                            e=e,
                                                            b=b,
                                                            TYPE=TYPE)

            return samples_and_pmf

        # generating_samples_func()
        samples_and_pmf = read_file_else_store(sample_file_path,
                                               generating_samples_func)
        samples = samples_and_pmf['splits_q_emp']
        pmf_q = samples_and_pmf['q']
        list_of_samples.append(samples)
        list_of_pmf_q.append(pmf_q)

    return list_of_samples, list_of_pmf_q


def load_generative_model_samples(power_base,
                                  ratio,
                                  dataset_type,
                                  m=10000,
                                  using_max=False):
    U = power_base**power_base
    num_s = 2
    num_resample = int(np.log2(ratio + 1)) - 1
    helper_database = SyntheticDataset(S=power_base,
                                       K=power_base,
                                       dataset_name=dataset_type,
                                       num_resample=num_resample,
                                       dont_generate=True)

    ground_truth_dict = make_stair_prob(U,
                                        posU=helper_database.U_pos/U,
                                        ratio=ratio,
                                        num_s=num_s)

    print(helper_database.p_likely)
    print(helper_database.p_rare)
    print(ground_truth_dict[0]['p'])
    print(ground_truth_dict[1]['p'])

    ground_truth_samples_list = generate_samples_scalable(
        ground_truth_dict, 10, U, m, tempered=False, e=0,
        b=100)['splits_q_emp']
    zero_space = 0
    for key, val in ground_truth_samples_list[0].items():
        if key > math.factorial(power_base):
            zero_space += 1
    pickle_files = ['sample_10000.pk']

    base_path = '{}_{}'.format(dataset_type, num_resample)
    dim_path = 'S_%d_K_%d' % (power_base, power_base)
    complete_path = os.path.join('checkpoints', base_path, dim_path)
    list_models = os.listdir(complete_path)
    samples_dict = {}

    for model in list_models:
        samples = []
        model_path = os.path.join(complete_path, model)
        list_models_date = os.listdir(model_path)
        list_models_date.sort()
        date_model = list_models_date[-1]
        model_path = os.path.join(model_path, date_model)
        model_path = os.path.join(model_path, 'figure')
        for file_name_pickle in pickle_files:
            samples_from_file = read_pickle_file(file_name_pickle,
                                                 model_path)['samples']
            empirical_dict = samples_to_histo(samples_from_file)
            empirical_dict = convert_key_sequence_to_int(
                power_base, empirical_dict, sequence_to_id, dataset_type)
            samples.append(empirical_dict)
        samples_dict[model] = samples
    samples_dict['ground truth'] = ground_truth_samples_list
    return samples_dict, ground_truth_dict
