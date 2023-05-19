import random
import math
from eval_src.basic_pmf import prob_dict_to_array
import numpy as np


def find_flat_regions(ground_truth_p_dict, is_optimized):
    if is_optimized:
        flat_regions = {}
        for key, val in ground_truth_p_dict.items():
            p_val = val['p']
            interval = val['interval']
            flat_regions[p_val] = range(interval[0], interval[1])

        return flat_regions
    else:
        flat_regions = {}
        for key, p_val in ground_truth_p_dict.items():
            if p_val in flat_regions:
                flat_regions[p_val].append(key)
            else:
                flat_regions[p_val] = [key]
        # sort all flat regions
        for p_val in flat_regions.keys():
            flat_regions[p_val].sort()
        return flat_regions


def binning_on_samples(consolidated_samples, trials, ground_truth_p_dict, B,
                       pmf_q):
    list_binned = []
    for trial in range(trials):
        result_dict = p_to_bp_algo(ground_truth_p_dict,
                                   consolidated_samples,
                                   B,
                                   seed=trial,
                                   pmf_q=pmf_q)
        # here its ok to transform the distribution to an array because the space is supppose to be very small

        binnned_p_hist = result_dict['new_histo_p']
        binnned_q_hist = result_dict['new_emp_histo_q']
        binnned_p_array = prob_dict_to_array(binnned_p_hist, B)
        binnned_q_array = prob_dict_to_array(binnned_q_hist, B)
        binned_dict = {'p': binnned_p_array, 'q': binnned_q_array}
        if pmf_q is not None:
            new_histo_q = result_dict['new_histo_q']
            binnned_true_q_array = prob_dict_to_array(new_histo_q, B)
            binned_dict['q_true'] = binnned_true_q_array

        list_binned.append(binned_dict)
    return list_binned


def find_flat_regions(ground_truth_p_dict, is_optimized):
    if is_optimized:
        flat_regions = {}
        for key, val in ground_truth_p_dict.items():
            p_val = val['p']
            interval = val['interval']
            flat_regions[p_val] = range(interval[0], interval[1])

        return flat_regions
    else:
        flat_regions = {}
        for key, p_val in ground_truth_p_dict.items():
            if p_val in flat_regions:
                flat_regions[p_val].append(key)
            else:
                flat_regions[p_val] = [key]
        # sort all flat regions
        for p_val in flat_regions.keys():
            flat_regions[p_val].sort()
        return flat_regions


def p_to_bp_algo(ground_truth_p_dict, q_dict, B, seed, pmf_q):
    random.seed(seed)  # this is to keep randomnes
    is_optimized = type(list(ground_truth_p_dict.values())[0]) is dict

    # 1 : find the S partitioning. By default, the zero space is "assumed" but not computed here.
    regions = find_flat_regions(ground_truth_p_dict, is_optimized)

    # 2 : Find each optimal split in each S regions. If the error is all pos or neg, there is no optimal split.

    S = len(regions) + 1

    # 4 : Return B* : mapping_bin_to_index is {index_bin: [all x], ...}
    mapping_bin_to_index = {}
    if B < S:  # NOT EXACT SOL
        raise NotImplemented
    # easiest scenario k=S, we just return all flat regions without cutting
    elif B == S:

        # S-1 because the zero region is implied
        for bin_ind, region_indices in enumerate(regions.values()):
            mapping_bin_to_index[bin_ind] = region_indices

    elif B > S:
        bin_ind = 0
        num_random_cut = math.floor(B - S)

        for bin_ind, region_indices in enumerate(regions.values()):
            mapping_bin_to_index[bin_ind] = region_indices

        num_random_cute_candidate = len(mapping_bin_to_index)
        random_cuts_per_bin = [0 for _ in range(num_random_cute_candidate)]

        for _ in range(num_random_cut):
            index = random.randint(0, num_random_cute_candidate - 1)
            random_cuts_per_bin[index] += 1

        for bin_id_to_cut, num_random_cuts in enumerate(random_cuts_per_bin):
            if num_random_cuts > 0:
                # if num_random_cuts>1:
                # print('problem')
                indices_to_split = mapping_bin_to_index[bin_id_to_cut]
                chunk_size = int(len(indices_to_split) / (num_random_cuts + 1))
                # first, we replace the bin by a small chunk of the indices:
                mapping_bin_to_index[
                    bin_id_to_cut] = indices_to_split[:chunk_size]
                # then we create news bins with all the chunks
                chunk_id = 0

                bin_ind += 1
                for chunk_id in range(1, num_random_cuts):
                    mapping_bin_to_index[bin_ind] = indices_to_split[
                        chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
                    bin_ind += 1

                mapping_bin_to_index[bin_ind] = indices_to_split[(chunk_id +
                                                                  1) *
                                                                 chunk_size:]

    new_histo_p = {}
    if pmf_q is not None:
        new_histo_q = {}

    for bin_index, all_index in mapping_bin_to_index.items():
        interval = (all_index[0], all_index[-1] + 1)
        new_probability_for_bin = get_summed_probabilities_of_interval(
            interval, ground_truth_p_dict, is_exact_interval=True)

        new_histo_p[bin_index] = new_probability_for_bin
        if pmf_q is not None:
            new_probability_for_bin = get_summed_probabilities_of_interval(
                interval, pmf_q)
            new_histo_q[bin_index] = new_probability_for_bin

    new_emp_histo_q = {}

    for bin_index, all_index in mapping_bin_to_index.items():
        interval = (all_index[0], all_index[-1] + 1)
        new_probability_for_bin = get_summed_probabilities_of_interval(
            interval, q_dict)
        new_emp_histo_q[bin_index] = new_probability_for_bin

    # add the zero bin
    new_emp_histo_q[B - 1] = 1 - np.sum(list(new_emp_histo_q.values()))
    new_histo_p[B - 1] = 0
    if pmf_q is not None:
        new_histo_q[B - 1] = 0
        return {
            'new_histo_p': new_histo_p,
            'new_emp_histo_q': new_emp_histo_q,
            'new_histo_q': new_histo_q
        }
    return {'new_histo_p': new_histo_p, 'new_emp_histo_q': new_emp_histo_q}


def get_overaps(sets_1, set_2):
    reminder = sets_1
    overlap = None
    start_b = set_2[0]
    end_b = set_2[1]
    for set in sets_1:
        if start_b >= set[0] and start_b < set[1]:  # {[ }
            if end_b > set[1]:  # {[ }  ],  {  [ }  ]
                overlap = [start_b, set[1]]
                reminder.remove(set)
                if start_b > set[0]:
                    reminder.append([set[0], start_b])
            else:  # {[   ]}  , {  [   ]  }  {[   ]  }  {  [   ]}
                overlap = [start_b, end_b]
                reminder.remove(set)
                if end_b < set[1]:  # {[   ]  }
                    reminder.append([end_b, set[1]])
                if start_b > set[0]:  # {  [   ]}
                    reminder.append([set[0], start_b])
        elif start_b < set[0] and end_b > set[0]:  # [ { ]
            if set[0] <= end_b:  # [ {   }]
                overlap = [set[0], set[1]]
                reminder.remove(set)

            else:  # [ { ] }
                overlap = [set[0], end_b]
                reminder.remove(set)
                reminder.append([end_b, set[1]])

    return overlap, reminder
def find_in_sorted_intervals(query, sorted_intervals):
    #current_list_intervals = sorted_intervals
    current_list_index = range(len(sorted_intervals))
    while True:
        # check if we have tiny list
        if len(current_list_index) == 0:
            return -1
        else:
            half_lookup = int(len(current_list_index)/2)
            current_index = current_list_index[half_lookup]
            current_interval = sorted_intervals[current_index]

            if current_interval[0] <= query and current_interval[1] > query:
                return current_index
            elif current_interval[1] <= query:
                current_list_index = current_list_index[half_lookup+1:]

            else:
                current_list_index = current_list_index[:half_lookup]


def find_interval(query, intervals):

    index_sorted = find_in_sorted_intervals(query, intervals)
    return index_sorted


def get_probability_at_element(j, histogram):
    is_optimized = type(list(histogram.values())[0]) is dict

    if is_optimized:
        intervals = [val['interval'] for _, val in histogram.items()]
        interval_index = find_interval(j, intervals)
        if interval_index in histogram:
            return histogram[interval_index]['p']
    else:
        if j in histogram:
            return histogram[j]
    return 0  # if the index doesn't appear, we assume the prob is 0


def get_summed_probabilities_of_interval(interval,
                                         histogram,
                                         is_exact_interval=False):
    is_optimized = type(list(histogram.values())[0]) is dict
    start = interval[0]
    end = interval[1]
    if end < start:
        print('problem')
    if is_exact_interval:
        probability = get_probability_at_element(start, histogram)
        summed_prob = probability * (end - start)
    else:  # we have to go through all keys
        if is_optimized:
            summed_prob = 0
            remaining_intervals = [interval]
            for key, val in histogram.items():
                overlap, remaining_intervals = get_overaps(
                    remaining_intervals, val['interval'])
                if overlap is not None:
                    size_interval = overlap[1] - overlap[0]
                    if size_interval < 0:
                        print('problem')
                    summed_prob += size_interval * val['p']
                if len(remaining_intervals) == 0:
                    return summed_prob

        else:
            summed_prob = 0
            for key, val in histogram.items():
                if key >= start and key < end:
                    summed_prob += val

    return summed_prob  # we always only have one prob per region