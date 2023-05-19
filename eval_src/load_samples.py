from cmath import exp
from src.datasets.real import RealDataset
from src.datasets.synthetic import SyntheticDataset
from eval_src.sampling import generate_samples_scalable
from eval_src.sequence_converter import convert_key_sequence_to_int, sequence_to_id
from eval_src.stair_p import make_stair_prob, samples_to_histo
from eval_src.file_helper import load_samples, read_pickle_file
import numpy as np
import os
import math


# This will either generate or load stored samples
def get_samples_p_q(experiment, experiment_config):

    if experiment == "SYNTH":

        ground_truth_p = make_stair_prob(experiment_config['U'],
                                         posU=experiment_config['U'],
                                         ratio=experiment_config['U'],
                                         num_s=experiment_config['U'])

        list_of_samples, list_of_pmf_q = load_samples(
            experiment_config['list_of_espilon_q'],
            experiment_config['init_b'], ground_truth_p,
            experiment_config['splits'], experiment_config['U'],
            experiment_config['m_per_splits'], experiment_config['num_s'],
            experiment_config['ratio'], experiment_config['TYPE'])

    elif experiment == "GEN":
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            experiment, experiment_config)
        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]
        list_of_pmf_q = None
    elif experiment == "PROXY":
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            experiment, experiment_config)

        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]
        list_of_pmf_q = None
    return list_of_samples, list_of_title_q, ground_truth_p, list_of_pmf_q


def load_generative_model_samples(experiment,
                                  experiment_config,
                                  using_max=False):
    if experiment == 'PROXY':
        helper_database = RealDataset(dataset_name='proxy', dont_generate=True)
        ground_truth_dict = helper_database.pmf
        pickle_files = ['sample_100000.pk']

        base_path = 'proxy'
        complete_path = os.path.join('checkpoints', base_path)
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
            try:
                for file_name_pickle in pickle_files:
                    samples_from_file = read_pickle_file(
                        file_name_pickle, model_path)['samples']

                    samples.append(samples_from_file)
                samples_dict[model] = samples
            except Exception as e:
                print('couldnt load {} ...'.format(model))
        experiment_config['m_per_splits'] = samples[0].shape[0]
        

    else:

        U = experiment_config['power_base']**experiment_config['power_base']
        assert U == experiment_config['U']
        m = experiment_config['m_per_splits']
        num_s = 2
        num_resample = int(np.log2(experiment_config['ratio'] + 1)) - 1
        helper_database = SyntheticDataset(
            S=experiment_config['power_base'],
            K=experiment_config['power_base'],
            dataset_name=experiment_config['TYPE'],
            num_resample=num_resample,
            dont_generate=True)

        ground_truth_dict = make_stair_prob(U,
                                            posU=helper_database.U_pos / U,
                                            ratio=experiment_config['ratio'],
                                            num_s=num_s)

        print(helper_database.p_likely)
        print(helper_database.p_rare)
        print(ground_truth_dict[0]['p'])
        print(ground_truth_dict[1]['p'])

        ground_truth_samples_list = generate_samples_scalable(
            ground_truth_dict, 10, m, 100000, tempered=False, e=0,
            b=100)['splits_q_emp']
        zero_space = 0
        for key, val in ground_truth_samples_list[0].items():
            if key > math.factorial(experiment_config['power_base']):
                zero_space += 1
        pickle_files = ['sample_100000.pk']

        base_path = '{}_{}'.format(experiment_config['TYPE'], num_resample)
        dim_path = 'S_%d_K_%d' % (experiment_config['power_base'],
                                  experiment_config['power_base'])
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
                    experiment_config['power_base'], empirical_dict,
                    sequence_to_id, experiment_config['TYPE'])
                samples.append(empirical_dict)
            samples_dict[model] = samples
        samples_dict['ground truth'] = ground_truth_samples_list
    return samples_dict, ground_truth_dict
