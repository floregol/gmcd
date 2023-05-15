def prob_dict_to_array(prob_hist, U):
    prob_array = []
    all_keys = list(prob_hist.keys())
    all_keys.sort()
    for key in range(U):
        if key in prob_hist:
            prob_array.append(prob_hist[key])
        else:
            prob_array.append(0)
    return prob_array


def get_pmf_val(key, pmf):
    all_intervals = list(pmf.values())

    for interval in all_intervals:
        if key >= interval['interval'][0] and key < interval['interval'][1]:
            return interval['p']
