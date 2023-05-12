from random import random
import numpy as np
import random 
import sympy
from scipy.stats import rv_discrete

from eval_src.tempering import generating_q_from_p


# sample a random index over a very large space without storing the whole space
def get_shuffled_index(i, base_b):
    a = 3
    def search_for_prime(b):
        current = b
        no_prime_found = True
        while no_prime_found:
            if sympy.isprime(current):
                return current
            current += 1
    base = search_for_prime(base_b)

    index_to_remove = [(i+(1))*a % base for i in range(base_b, base)]

    index = (i+(1))*a % base
    off_set = np.sum(np.array(index_to_remove) < index)
    index = index - off_set
    assert index < base_b
    return index


def sample_pmf(value, probability, m):
    distrib = rv_discrete(values=(value, probability))
    new_samples = distrib.rvs(size=m)
    return new_samples


def scalabale_sample_distribution_with_shuffle(prob_optimized_dict, ground_truth_p, m):
    mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                         for key, val in prob_optimized_dict.items()]
    should_be_one = np.sum(mass_in_each_part)
    size_each_regions = [(val['interval'][1] - val['interval'][0])
                         for _, val in prob_optimized_dict.items()]
    regions_to_be_merges = {}
    for key, val in prob_optimized_dict.items():
        interval_to_place = val['interval']
        for key_bigger, val in ground_truth_p.items():
            bigger_int = val['interval']
            if interval_to_place[0] >= bigger_int[0] and interval_to_place[1] <= bigger_int[1]:
                regions_to_be_merges[key] = key_bigger

    regions = list(prob_optimized_dict.keys())
    # FIRST SAMPLING, which region to sample from
    samples = sample_pmf(regions, mass_in_each_part, m)
    index_with_regions = []
    for region in regions:
        # small error, the first number can be overwritten
        index_with_region = np.where(samples == region)[0]
        index_with_regions.append(index_with_region)
    for i, region in enumerate(regions):
        index_with_region = index_with_regions[i]
        m_in_region = index_with_region.shape[0]
        if m_in_region > 0:
            interval_of_region = prob_optimized_dict[region]['interval']
            size_region = interval_of_region[1] - interval_of_region[0]
            if i in regions_to_be_merges:
                base_interval = ground_truth_p[regions_to_be_merges[i]]['interval']
                size_base_region = base_interval[1] - base_interval[0]
                offset = interval_of_region[0] - base_interval[0]
                in_region_samples = random.choices(
                    range(offset, offset+size_region), k=m_in_region)
                print('shuffling process within the samples')
                base_offset = base_interval[0]
                shuffled_index = [get_shuffled_index(
                    s, base_b=size_base_region)+base_offset for s in in_region_samples]

                samples[index_with_region] = shuffled_index
            else:  # zero space, no shuffling needed
                in_region_samples = random.choices(
                    range(interval_of_region[0], interval_of_region[1]), k=m_in_region)
                samples[index_with_region] = in_region_samples
    return samples

def empirical_dist_no_zero(incoming_m, incoming_array_samples):
    p_emp = {}
    for sample in incoming_array_samples:
        if sample in p_emp:
            p_emp[sample] += 1/incoming_m
        else:
            p_emp[sample] = 1/incoming_m

    return p_emp

def generate_samples_scalable(ground_truth_p, splits, U, m, tempered, e, b, TYPE=None):
    splits_q_emp = []
    percent_to_modify_null = 0.1
    print('PERCENT NULL', percent_to_modify_null)
    # first, check if the ground truth is given in the optimized format
    
    prob_optimized_dict = ground_truth_p
    if tempered:
        prob_optimized_dict = generating_q_from_p(
            U, ground_truth_p, e, b, percent_to_modify_null, TYPE=TYPE)
    q = prob_optimized_dict
    for _ in range(splits):

        new_samples = scalabale_sample_distribution_with_shuffle(
            prob_optimized_dict, ground_truth_p, m)
        p_emp_dict = empirical_dist_no_zero(m, new_samples)
        splits_q_emp.append(p_emp_dict)
    return {'splits_q_emp': splits_q_emp, 'q': q}