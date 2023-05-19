
from eval_src.file_helper import create_prefix_from_list, store_for_plotting
import numpy as np
import random
from eval_src.generate_omegadelta import convert_to_pattern, preprocess_pmf_p
from eval_src.load_samples import get_samples_p_q
from eval_src.metric_p import compute_NLL, perform_our_test
from eval_src.table_helper import build_latex_table
# from table_helper import build_latex_table


def float_to_print(number, num_d=3):
    if num_d == 3:
        return '{:.3f}'.format(number)
    elif num_d == 2:
        return '{:.2f}'.format(number)
    elif num_d == 1:
        return '{:.1f}'.format(number)


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)

    experiment = "PROXY"  # either SYNTH or GEN o PROXY

    experiment_config = {}
    delta = 0.05
    trials = 50

    if experiment == 'PROXY':
        U = 21**53
        experiment_config['K'] = 21
        experiment_config['S'] = 53
    else:
        power_base = 6
        U = power_base**power_base
        experiment_config['power_base'] = power_base
        experiment_config['K'] = power_base
        experiment_config['S'] = power_base

    experiment_config['trials'] = trials
    experiment_config['U'] = U
    experiment_config['delta'] = delta

    if experiment == "SYNTH":  # if we generate q ourselves
        print('You are running the synthetic experiment...')
        TYPE = "TAIL"  # TAIL, SHARP, FLAT
        distribution_type = 'STAIRS'  # STAIRS
        posU = 0.9
        m_per_splits = 10000
        init_e = 0.05
        init_b = 0.3
        splits = 10
        num_s = 4
        ratio = 5
        list_of_espilon_q = [0, init_e, init_e * 1.5, init_e * 2]
        list_of_title_q = [
            TYPE + ':q ' + float_to_print(e) for e in list_of_espilon_q
        ]
        experiment_config['TYPE'] = TYPE
        experiment_config['distribution_type'] = distribution_type
        experiment_config['posU'] = posU
        experiment_config['m_per_splits'] = m_per_splits
        experiment_config['init_b'] = init_b
        experiment_config['splits'] = splits
        experiment_config['num_s'] = num_s
        experiment_config['ratio'] = ratio

    elif experiment == "GEN":  # if we take q as the generative models we have, we load the samples.
        print('You are running the generative model experiment...')
        TYPE = 'sort'
        m_per_splits = 100000
        num_s = 2
        ratio = 7  # 7d also 3
        splits = 10
        experiment_config['TYPE'] = TYPE
        experiment_config[
            'm_per_splits'] = m_per_splits  # TODO remove as an arg
        experiment_config['splits'] = splits
        experiment_config['num_s'] = num_s
        experiment_config['ratio'] = ratio

    elif experiment == "PROXY":  # if we take q as the generative models we have, we load the samples.
        print('You are running the PROXY model experiment...')
        splits = 10
        TYPE = 'proxy'
        experiment_config['TYPE'] = TYPE
        experiment_config['splits'] = splits

    list_of_samples, list_of_title_q, ground_truth_p, list_of_pmf_q = get_samples_p_q(
        experiment, experiment_config)

    print("for this round m is ", experiment_config['m_per_splits'] * splits)
    print("and U is ", U)
    metrics = ['S', 'test', 'binning', 'A', 'nll', 'e', 'std_nll', 'l1']
    store_results = {}
    store_results_ranking = {}

    for algo in ['algo']:
        store_results_ranking[algo] = []
    for metric in metrics:
        store_results[metric] = {}
        for title in list_of_title_q:
            store_results[metric][title] = {}
    ground_truth_samples = list_of_samples[0]

    if experiment == "PROXY":
        omegaDelta = preprocess_pmf_p(ground_truth_p, num_s=6)
        experiment_config['num_s'] = 6
        list_of_samples, ground_truth_p = convert_to_pattern(
            omegaDelta, list_of_samples)
        #experiment_config['num_s'] = len(ground_truth_p)

    perform_our_test(list_of_samples, list_of_title_q, store_results,
                     ground_truth_p, list_of_pmf_q, experiment_config)

    if list_of_pmf_q is not None:
        compute_NLL(ground_truth_samples, list_of_pmf_q, list_of_title_q,
                    store_results, m_per_splits)

    #coverage_baselines(ground_truth_samples, list_of_samples)
    if experiment == "SYNTH":
        prefix = create_prefix_from_list({
            'exp': experiment + TYPE,
            'U': U,
            'm_per_splits': m_per_splits,
            'splits': splits,
            'S': num_s,
            'ratio': ratio,
            'b': init_b,
            'e': init_e
        })
    else:
        prefix = create_prefix_from_list({
            'exp': experiment + TYPE,
            'U': U,
            'm_per_splits': m_per_splits,
            'splits': splits,
            'S': num_s,
            'ratio': ratio
        })
    store_for_plotting(data={'data': store_results}, title=prefix)

    rows = []
    for q_name in list_of_title_q:
        values = []
        if list_of_pmf_q is not None:
            values = [float_to_print(np.mean(store_results['nll'][q_name]))]

        for key, val in store_results['A'][q_name].items():
            std = np.mean((store_results['e'][q_name][key]))
            values.append(float_to_print(np.mean(val), num_d=3))
        rows.append([q_name] + values)
    top = ['']
    if list_of_pmf_q is not None:
        top = top + ['nll']
    for B in store_results['A'][q_name].keys():
        top = top + ['$B_' + str(B) + '$']
        #top = top + [ '$tv$']
    build_latex_table([top] + rows,
                      caption=TYPE + ' m/Omega' + float_to_print(
                          (m_per_splits * splits) / U) + ' S:' + str(num_s),
                      label=prefix)
