import numpy as np
import scipy
from sklearn.covariance import log_likelihood
from eval_src.file_helper import create_prefix_from_list, load_samples, store_for_plotting
# from eval_src.sampling.loading_samples import load_generative_model_samples
# from eval_src.sampling.stair import make_stair_prob
# from eval_src.sampling.discrete import makeUniProbArr, prob_array_to_dict
# from eval_src.statistic.binning_algo import binning_on_samples
# from eval_src.statistic.generate_statistics import compute_norm, genSstat, get_pmf_val, get_ranking_results, reject_if_bad_test
import numpy as np
import random
import math
from tqdm import tqdm

from eval_src.stair_p import make_stair_prob
# from table_helper import build_latex_table


def float_to_print(number, num_d=3):
    if num_d == 3:
        return '{:.3f}'.format(number)
    elif num_d ==2:
        return  '{:.2f}'.format(number)
    elif num_d ==1:
        return  '{:.1f}'.format(number)

if __name__ == '__main__':
    # Set the random seed
    np.random.seed(3)
    random.seed(3)
    
    experiment = "SYNTH"  # either SYNTH or GEN
    
    
    test_epsilon = None
    delta = 0.5
    power_base = 10
    U = power_base**power_base
    trials = 50

    list_of_binning = ['algo']
    if experiment == "SYNTH":  # if we generate q ourselves
        print('You are running the synthetic experiment...')
        TYPE = "FLAT"  # TAIL, SHARP, FLAT
        posU = 0.9
        m_per_splits = 5000
        init_e = 0.05
        init_b = 0.3
        splits = 10
        num_s = 4
        ratio = 5
        distribution_type = 'STAIRS'  # STAIRS
        list_of_espilon_q = [0, init_e, init_e*1.5, init_e*2]
        list_of_title_q = [TYPE+':q ' +
                           float_to_print(e) for e in list_of_espilon_q]

    else:  # if we take q as the generative models we have, we load the samples.
        print('You are running the generative model experiment...')
       
        m_per_splits = 10000
        S = 2
        ratio = 3
        splits = 10

    print("for this round m is ", m_per_splits*splits)
    print("and U is ", U)
    metrics = ['S', 'test', 'binning', 'A', 'nll', 'e', 'std_nll', 'l1']
    if experiment == "SYNTH":
        
        
        
        ground_truth_p = make_stair_prob(U, posU=posU, ratio=ratio,  num_s=num_s)

        
        list_of_samples, list_of_pmf_q = load_samples(
            list_of_espilon_q, init_b, ground_truth_p, splits, U, m_per_splits,num_s, ratio, TYPE)
        store_results = {}
        store_results_ranking = {}
        for algo in list_of_binning:
            store_results_ranking[algo] = []

        for metric in metrics:
            store_results[metric] = {}
            for title in list_of_title_q:
                store_results[metric][title] = {}
    else:
        dict_of_samples, ground_truth_p = load_generative_model_samples(
            power_base, num_files=10)
        list_of_samples = [val for _, val in dict_of_samples.items()]
        list_of_title_q = [key for key, _ in dict_of_samples.items()]
        store_results = {}
        store_results_ranking = {}
        for algo in list_of_binning:
            store_results_ranking[algo] = []
        for metric in metrics:
            store_results[metric] = {}
            for title in list_of_title_q:
                store_results[metric][title] = {}
        list_of_pmf_q = None
    
    perform_our_test(list_of_samples, list_of_title_q,
                     S, trials, store_results, list_of_pmf_q)
    ground_truth_samples = list_of_samples[0]
    if list_of_pmf_q is not None:
        compute_NLL(ground_truth_samples, list_of_pmf_q,
                    list_of_title_q, store_results)

    #coverage_baselines(ground_truth_samples, list_of_samples)
    if experiment == "SYNTH":
        prefix = create_prefix_from_list(
            {'exp': experiment+TYPE, 'U': U, 'm_per_splits': m_per_splits, 'splits': splits, 'S': S, 'ratio': ratio, 'b': init_b, 'e': init_e})
    else:
        prefix = create_prefix_from_list(
            {'exp': experiment+TYPE, 'U': U, 'm_per_splits': m_per_splits, 'splits': splits, 'S': S, 'ratio': ratio})
    store_for_plotting(data={'data': store_results}, title=prefix)

    rows = []
    for q_name in list_of_title_q:
        values = []
        if list_of_pmf_q is not None:
            values = [float_to_print(np.mean(store_results['nll'][q_name])) ]
        
        for key, val in store_results['A'][q_name].items():
            std = np.mean((store_results['e'][q_name][key]))
            values.append(float_to_print(np.mean(val),num_d=3) )
        rows.append([q_name] + values)
    top = ['']
    if list_of_pmf_q is not None:
        top = top + ['nll']
    for B in store_results['A'][q_name].keys():
        top = top + ['$B_'+str(B)+'$']
        #top = top + [ '$tv$']
    build_latex_table([top]+rows, caption=TYPE + ' m/Omega' +
                      float_to_print((m_per_splits*splits)/U) + ' S:'+str(S), label=prefix)
