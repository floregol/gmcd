from eval_src.stair_p import make_stair_prob
from eval_src.file_helper import create_prefix_from_list, load_generative_model_samples, load_samples, store_for_plotting

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

    else:
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            experiment_config['power_base'], experiment_config['ratio'],
            experiment_config['TYPE'])
        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]
        list_of_pmf_q = None
    return list_of_samples, list_of_title_q, ground_truth_p, list_of_pmf_q
