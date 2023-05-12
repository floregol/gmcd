import numpy as np
from tqdm import tqdm

# basic binary search


def find_in_sorted_intervals(query, sorted_intervals):
    #current_list_intervals = sorted_intervals
    current_list_index = range(len(sorted_intervals))
    while True:
        # check if we have tiny list
        if len(current_list_index) == 0:
            return -1
        else:
            half_lookup = int(len(current_list_index) / 2)
            current_index = current_list_index[half_lookup]
            current_interval = sorted_intervals[current_index]

            if current_interval[0] <= query and current_interval[1] > query:
                return current_index
            elif current_interval[1] <= query:
                current_list_index = current_list_index[half_lookup + 1:]

            else:
                current_list_index = current_list_index[:half_lookup]


def find_interval(query, intervals):

    index_sorted = find_in_sorted_intervals(query, intervals)
    return index_sorted


def get_overap(interval_M, interval_remain):  # return remainder from 2
    start_M = interval_M[0]
    end_M = interval_M[1]
    start_R = interval_remain[0]
    end_R = interval_remain[1]
    if end_R <= start_M or end_M <= start_R:  # no overlap
        return None, [], []
    elif start_M <= start_R:  # [0,..] [5,...]

        new_interval = (start_R, end_M)
        if end_M < end_R:
            post_remain = (end_M, end_R)
            return new_interval, [], [post_remain]
        else:
            return new_interval, [], []
    else:  # start_M > start_R
        new_interval = (start_M, end_R)
        pre_remain = (start_R, start_M)
        if end_M < end_R:
            post_remain = (end_M, end_R)
            return new_interval, [pre_remain], [post_remain]
        else:
            return new_interval, [pre_remain], []


def generating_q_from_p(U, prob_optimized_dict, e, percent_of_space_to_modify,
                        percent_to_modify_null, TYPE):

    percent_pos_space = percent_of_space_to_modify - percent_to_modify_null
    if TYPE == 'TAIL':
        # if we only want to modify the tail, we cant mess the whole space
        assert percent_pos_space < 0.5
    assert percent_pos_space >= 0

    mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                         for _, val in prob_optimized_dict.items()]
    should_be_one = np.sum(mass_in_each_part)
    size_each_regions = [(val['interval'][1] - val['interval'][0])
                         for _, val in prob_optimized_dict.items()]
    intervals = [val['interval'] for _, val in prob_optimized_dict.items()]

    U_pos = np.sum(size_each_regions)  # count the positive space

    e_per_section = e / 2
    bins_added = int(U * percent_of_space_to_modify / 2)
    bins_removed = int(U * percent_of_space_to_modify / 2)
    bins_added_in_null = int(bins_added * percent_to_modify_null)
    bins_added_in_pos = bins_added - bins_added_in_null

    e_added = e_per_section / bins_added  # error amount to add per element
    e_removed = e_per_section / bins_removed  # error amount to subtract per element
    print('Total tv error should be ', e, ' it is ',
          e_added * bins_added + e_removed * bins_removed)
    print('Total l2^2 error is ',
          (e_added**2) * bins_added + (e_removed**2) * bins_removed)
    print('Total l2 error is ',
          np.sqrt((e_added**2) * bins_added + (e_removed**2) * bins_removed))
    """
    modification in the positive space
    """

    print('Starting the tempering process... Less randomized but way faster')
    new_inverse_tempered_dict = {}
    untouched_intervals = list(range(len(intervals)))
    if TYPE == 'SHARP' or TYPE == 'FLAT':
        # positive tempering
        num_per_interval = int((len(intervals)) / 2)
        if TYPE == 'SHARP':
            num_to_modify_pre_interval = int(bins_added_in_pos /
                                             num_per_interval)
            num_to_modify_post_interval = int(bins_removed / num_per_interval)
        else:
            num_to_modify_pre_interval = int(bins_removed / num_per_interval)
            num_to_modify_post_interval = int(bins_added_in_pos /
                                              num_per_interval)

        for i in range(num_per_interval):
            untouched_intervals.remove(i)
            interval = intervals[i]
            interval_to_modify = (interval[0],
                                  interval[0] + num_to_modify_pre_interval)
            new_interval, _, interval_remains_post = get_overap(
                interval_to_modify, interval_remain=interval)
            p_value_of_interval = prob_optimized_dict[i]['p']
            if TYPE == 'SHARP':
                new_p_value = p_value_of_interval + e_added
            else:
                new_p_value = p_value_of_interval - e_removed
            new_inverse_tempered_dict[new_p_value] = [new_interval]
            new_inverse_tempered_dict[
                p_value_of_interval] = interval_remains_post
        # negative tempering
        for i in range(num_per_interval):
            reverse_index = len(intervals) - 1 - i
            untouched_intervals.remove(reverse_index)
            interval = intervals[reverse_index]
            interval_to_modify = (interval[0],
                                  interval[0] + num_to_modify_post_interval)
            new_interval, _, interval_remains_post = get_overap(
                interval_to_modify, interval_remain=interval)
            p_value_of_interval = prob_optimized_dict[reverse_index]['p']
            if TYPE == 'SHARP':
                new_p_value = p_value_of_interval - e_removed
            else:
                new_p_value = p_value_of_interval + e_added
            new_inverse_tempered_dict[new_p_value] = [new_interval]
            new_inverse_tempered_dict[
                p_value_of_interval] = interval_remains_post

    elif TYPE == 'ANOM' or TYPE == 'UNI' or TYPE == 'TAIL':
        # for each interval, divide it in two and add to the start, remove from lasts.
        intervals_modified = intervals
        if TYPE == 'TAIL':
            bins_added = int(bins_added / 3)
            coin = e * 100 % 2
            limit_tail_index = int(len(intervals) / 2)
            if coin == 1:
                intervals_modified = intervals[limit_tail_index:]
            else:
                intervals_modified = intervals[:limit_tail_index]
                limit_tail_index = 0
        e_added_pos = e_added
        if TYPE == 'ANOM':
            bins_added = int(bins_added / 4)
        e_in_zero = bins_added_in_null * e_added
        e_added_pos = (e_per_section - e_in_zero) / bins_added

        num_to_add_per_interval = bins_added / len(intervals_modified)
        remainder_add = bins_added % len(intervals_modified)
        num_to_remove_per_interval = bins_removed / len(intervals_modified)

        remainder_remove = bins_removed % len(intervals_modified)

        num_to_add_per_interval = int(num_to_add_per_interval)
        num_to_remove_per_interval = int(num_to_remove_per_interval)
        for i, interval in enumerate(intervals_modified):
            if i == len(intervals_modified
                        ) - 1:  # last one, we add the remainder here
                num_to_add_per_interval = num_to_add_per_interval + remainder_add
                num_to_remove_per_interval = num_to_remove_per_interval + remainder_remove
            if TYPE == 'TAIL':
                i = i + limit_tail_index
            untouched_intervals.remove(i)
            p_value_of_interval = prob_optimized_dict[i]['p']
            interval_to_add = (interval[0],
                               interval[0] + num_to_add_per_interval)
            new_interval_add, _, interval_remains_post = get_overap(
                interval_to_add, interval_remain=interval)
            interval_to_remove = (interval[1] - num_to_remove_per_interval,
                                  interval[1])
            new_interval_remove, interval_remains, _ = get_overap(
                interval_to_remove, interval_remain=interval_remains_post[0])

            new_inverse_tempered_dict[p_value_of_interval +
                                      e_added_pos] = [new_interval_add]
            new_inverse_tempered_dict[p_value_of_interval -
                                      e_removed] = [new_interval_remove]
            new_inverse_tempered_dict[p_value_of_interval] = interval_remains

    for i in untouched_intervals:
        p_value_of_interval = prob_optimized_dict[i]['p']
        interval = intervals[i]
        new_inverse_tempered_dict[p_value_of_interval] = [interval]
    print('starting the inverting process...')
    # invert the dict
    new_tempered_dict = {}
    j = 0
    for key, val in tqdm(new_inverse_tempered_dict.items()):
        sorted_val = sorted(val, key=lambda x: x[0])
        for interval in sorted_val:
            new_tempered_dict[j] = {'interval': interval, 'p': key}
            j += 1
    """
    modification in the null/zero space
    """
    assert e_added not in new_tempered_dict  # hoping
    mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                         for key, val in new_tempered_dict.items()]
    should_be_one = np.sum(mass_in_each_part)
    new_tempered_dict[j] = {
        'interval': (U_pos, U_pos + bins_added_in_null),
        'p': e_added
    }
    # We just add in order in the null space
    mass_in_each_part = [(val['interval'][1] - val['interval'][0]) * val['p']
                         for key, val in new_tempered_dict.items()]
    should_be_one = np.sum(mass_in_each_part)
    np.testing.assert_allclose(should_be_one, 1)
    print('final verif..')
    error = 0
    for key, val in new_tempered_dict.items():
        size = val['interval'][1] - val['interval'][0]
        start = val['interval'][0]
        index_in_init_p = find_interval(
            start, [val['interval'] for _, val in prob_optimized_dict.items()])

        new_p = val['p']
        old_p = 0
        if index_in_init_p >= 0:
            old_p = prob_optimized_dict[index_in_init_p]['p']

        error += np.abs(new_p * size - old_p * size)
    print('Total tv error should be ', e, ' it is ', error)
    return new_tempered_dict