import math
import numpy as np

#from eval_src.tempering import generating_q_from_p


def samples_to_histo(samples):
    
    amount_samples = len(samples)
    increment = 1/amount_samples
    empirical_dict = {}
    for item in samples:
        # get value currently and add the "increment"
        if str(item) in empirical_dict:
            # gets the value corresponding to the key (item, here)
            val = empirical_dict.get(str(item))
            val += increment
            empirical_dict.update({str(item): val})
        else:
            # add item to empirical_dict with the base "increment"
            empirical_dict.update({str(item): increment})

    
    should_be_one = np.sum(list(empirical_dict.values()))
    print('should_be_one', should_be_one)
    return empirical_dict
def make_stair_prob(U, posU, ratio, num_s):
    U_with_stair = int(posU * U)

    U_for_each_S = math.floor(U_with_stair / num_s)
    U_for_last_S = U_with_stair - (num_s - 1) * U_for_each_S
    U_per_stairs = [U_for_each_S for i in range(num_s - 1)]
    U_per_stairs.append(U_for_last_S)

    ratio_all_steps = list(np.arange(1, ratio, (ratio - 1) / (num_s - 1)))
    ratio_all_steps.append(ratio)
    p_first_floor = 1 / \
        (np.sum([U_per_stairs[i]*ratio_step for i,
         ratio_step in enumerate(ratio_all_steps)]))
    p_each_stair = [
        p_first_floor * ratio_stair for ratio_stair in ratio_all_steps
    ]
    verify_that_is_one = np.sum(
        [p_each_stair[i] * U_per_stairs[i] for i in range(num_s)])

    U_per_stairs.reverse()
    p_each_stair.reverse()
    current_dist = 0
    stair_histo = {}
    start_interval = 0
    for i, size_stair in enumerate(U_per_stairs):
        current_dist = p_each_stair[i]
        interval = [start_interval, start_interval + size_stair]
        stair_histo[i] = {'interval': interval, 'p': current_dist}
        start_interval += size_stair
    should_sum_to_one = np.sum([(val['interval'][1] -val['interval'][0])*val['p']  for _, val in stair_histo.items()])
    return stair_histo


