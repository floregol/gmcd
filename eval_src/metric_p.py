import numpy as np
from tqdm import tqdm
import scipy
from eval_src.basic_pmf import get_pmf_val
from eval_src.binning import binning_on_samples


def compute_NLL(ground_truth_samples, list_of_pmf_q, list_of_title_q,
                store_results, m_per_splits):
    all_log_likelihoods = []
    for i, q_name in enumerate(list_of_title_q):
        log_likelihoods = []
        pmf = list_of_pmf_q[i]
        q_name = list_of_title_q[i]

        for trial in ground_truth_samples:
            log_likelihood = 0
            for key, val in trial.items():
                p_key = get_pmf_val(key, pmf)
                log_p = np.log(p_key)
                num_int = int(val * m_per_splits)
                log_likelihood += log_p * num_int
            log_likelihoods.append(-log_likelihood / m_per_splits)
        print(q_name, 'log likelihood m=', m_per_splits, ':',
              np.mean(log_likelihoods), 'std', np.std(log_likelihoods))
        all_log_likelihoods.append(log_likelihoods)
        store_results['nll'][q_name] = log_likelihoods
        store_results['std_nll'][q_name] = np.std(log_likelihoods)
    print(all_log_likelihoods[0])
    print(all_log_likelihoods[1])
    print(scipy.stats.wilcoxon(all_log_likelihoods[0], all_log_likelihoods[2]))


def consolidate(all_samples_list):
    sample_dict = {}
    num_splits = len(all_samples_list)
    for samples in all_samples_list:
        for key, emp_q in samples.items():
            if key not in sample_dict:
                sample_dict[key] = emp_q / num_splits
            else:
                sample_dict[key] += emp_q / num_splits
    print('should be one', np.sum(list(sample_dict.values())))
    return sample_dict


def compute_norm(p_array, q_array):
    l2 = 0
    l1 = 0
    for i in range(len(p_array)):
        l1_x = np.abs(p_array[i] - q_array[i])
        l2_x = l1_x**2
        l2 += l2_x
        l1 += l1_x
    l2 = np.sqrt(l2)
    l1 = l1 / 2
    return {'l1': l1, 'l2': l2}


def reject_if_bad_test(prob_array, q_emp_array, m, delta=1 / 3):
    U = len(prob_array)
    empirical_dtv = 0
    for x in range(U):
        tv_x = np.abs(prob_array[x] - q_emp_array[x])
        empirical_dtv += tv_x
    empirical_dtv = empirical_dtv / 2
    e_test = max(np.sqrt(U / m), np.sqrt(2 / m * np.log2(2 / delta)))
    if empirical_dtv <= e_test:
        test_pass = True
    else:
        test_pass = False
    return {
        'close_enough': test_pass,
        'e_test': e_test,
        'emp_dtv': empirical_dtv
    }


def perform_our_test(list_of_samples, list_of_title_q, store_results,
                     ground_truth_p, list_of_pmf_q, experiment_config):

    # step one consolidate all samples to one sample set
    consolidated_samples = []
    for all_samples_list in list_of_samples:
        consolidated_samples.append(consolidate(all_samples_list))

    Bs = list(range(experiment_config['num_s'] + 1, 8))
    for B in tqdm(Bs):  # For each bin granularity

        for i, consolidated_samples_baseline in enumerate(
                consolidated_samples):
            pmf_q = None
            if list_of_pmf_q is not None:
                pmf_q = list_of_pmf_q[i]
            list_binned = binning_on_samples(consolidated_samples_baseline,
                                             experiment_config['trials'],
                                             ground_truth_p, B, pmf_q)
            # run statistical test
            results = [
                reject_if_bad_test(trial['p'],
                                   trial['q'],
                                   experiment_config['splits'] *
                                   experiment_config['m_per_splits'],
                                   delta=experiment_config['delta'])
                for trial in list_binned
            ]
            test = [i['close_enough'] for i in results]
            A = [i['emp_dtv'] for i in results]
            error = [i['e_test'] for i in results]

            q_name = list_of_title_q[i]
            if pmf_q is not None:
                true_norm_results = [
                    compute_norm(trial['p'], trial['q_true'])
                    for trial in list_binned
                ]
                l2 = [i['l2'] for i in true_norm_results]
                l1 = [i['l1'] for i in true_norm_results]
                store_results['l1'][q_name][B] = l1

            store_results['test'][q_name][B] = test
            store_results['A'][q_name][B] = A

            store_results['e'][q_name][B] = error
            store_results['binning'][q_name][B] = list_binned


