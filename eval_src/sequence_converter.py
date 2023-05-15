import itertools
import random
import math
import numpy as np


def get_closest_smaller_perm(sequence, base):
    x_dict = {}
    reminding = list(range(base))
    closest_smaller_permuation = []
    for x in sequence:
        if x in x_dict:
            usable_x = [r < x for r in reminding]
            index = int(np.sum(usable_x)) - 1
            if index < 0:  # go back
                for i in list(reversed(range(
                        len(closest_smaller_permuation)))):
                    xs = closest_smaller_permuation[i]
                    usable_x = [r < xs for r in reminding]
                    index = int(np.sum(usable_x)) - 1
                    if index >= 0:  # replace the end
                        closest_smaller_permuation = closest_smaller_permuation[:
                                                                                i]
                        closest_smaller_permuation.append(reminding[index])
                        for x in reversed(range(base)):
                            if x not in closest_smaller_permuation:
                                closest_smaller_permuation.append(x)
                        return closest_smaller_permuation
                    else:
                        reminding.append(xs)
                        reminding.sort()
                return None
            else:
                closest_smaller_permuation.append(reminding[index])
                if reminding[index] in reminding:
                    reminding.remove(reminding[index])
                closest_smaller_permuation = closest_smaller_permuation + \
                    list(reversed(reminding))
                return closest_smaller_permuation
        else:
            closest_smaller_permuation.append(x)
            x_dict[x] = 1
            reminding.remove(x)
    return closest_smaller_permuation


def count_permutation_before(sequence, base):
    # find closest smallest permutation before
    closest_smaller_permuation = get_closest_smaller_perm(sequence, base)
    if closest_smaller_permuation is not None:
        count_permutation_before = get_permuation_rank(
            closest_smaller_permuation, base) + 1
        return count_permutation_before
    else:
        return 0


def likely_with_start(b_sequence, base):
    remainding = list(range(base))
    for s in b_sequence:
        remainding.remove(s)
    can_end = np.sum([r > b_sequence[0] for r in remainding])
    likely_with_start = can_end * math.factorial(base - len(b_sequence) - 1)
    return likely_with_start


def to_permutation(b_sequence, base):
    d = {}
    s = []
    for b in b_sequence:
        if b not in d:
            s.append(b)
            d[b] = 1
        else:
            while b in d:
                b = b + 1

            if b < base and b >= 0:
                d[b] = 1
                s.append(b)
            else:
                return None

    return s


def get_type(comb, dataset_type, base):
    if dataset_type == 'sort':
        choices = {}
        for x in comb:
            if x not in choices:
                choices[x] = 1
            else:
                return 'not a permutation'
        if comb[0] < comb[-1]:
            return 'likely'
        else:
            return 'rare'
    elif dataset_type == 'pair':
        if comb[0] not in [i for i in range(base)]:
            return 'not a pair'
        for i, x in enumerate(comb[1:]):
            choices = [
                i % base for i in range(comb[i], comb[i] + int(base / 2))
            ]
            if x not in choices:
                return 'not a pair'
        if (comb[0] + comb[-1]) % 2 == 0:
            return 'likely'
        else:
            return 'rare'


def count_likely_before(sequence, base):
    start = sequence[0]
    # all likely that starts with a lower number will be before.
    # if start = 2, 0123, 0132, .... then 1032, 1302, ...
    num_likely_before = [(base - i - 1) * math.factorial(base - 2)
                         for i in range(start)]
    num_likely_before = int(np.sum(num_likely_before))
    # then all likely that starts with same number
    num_likely_at = (base - start - 1) * math.factorial(base - 2)

    for i in range(1, base - 1):
        begining_sequence = sequence[:i]
        begining_s = to_permutation(begining_sequence, base)
        if begining_s is not None:
            for b in range(sequence[i] + 1, base):
                b_sequence = begining_s + [b]
                if get_type(b_sequence, 'sort', base=base) != 'not a permutation':
                    inaccessible_likely = likely_with_start(b_sequence, base)
                    num_likely_at = num_likely_at - inaccessible_likely
    num_likely_at = num_likely_at - 1  # remove the sequence itself
    return num_likely_before + num_likely_at


def get_permuation_rank(sequence, base):
    seq_id = 0
    nums_left = list(range(base))
    for s, x in enumerate(sequence):
        offset = nums_left.index(x) * math.factorial(base - s - 1)
        nums_left.remove(x)
        nums_left.sort()
        seq_id += offset
    return seq_id


def get_sequence_rank(sequence, base):
    seq_id = 0
    # count in base
    for s, x in enumerate(sequence):
        seq_id += base**(base - s - 1) * x
    return seq_id


likely_dict = {}
likely_counter = 0
rare_dict = {}
rare_counter = 0
empty_dict = {}
empty_counter = 0

for sequence in itertools.product([i for i in range(6)], repeat=6):
    type_seq = get_type(sequence, 'pair', base=6)
    if type_seq == 'likely':
        likely_dict[sequence] = likely_counter
        likely_counter += 1
    elif type_seq == 'rare':
        rare_dict[sequence] = rare_counter
        rare_counter += 1
    else:
        empty_dict[sequence] = empty_counter
        empty_counter += 1


def brute_force_order(sequence, base, dataset_type):
    type_seq = get_type(sequence, dataset_type, base=base)
    x = tuple(sequence)
    if type_seq == 'likely':
        seq_id = likely_dict[x]
    elif type_seq == 'rare':
        seq_id = rare_dict[x] + len(likely_dict)
    else:
        seq_id = empty_dict[x] + len(rare_dict) + len(likely_dict)
    return seq_id


def sequence_to_id(sequence, base, dataset_type):
    if dataset_type == 'sort':
        seq_id = 0

        # print(nums_left)
        type_seq = get_type(sequence, dataset_type, base=base)
        if type_seq == 'likely' or type_seq == 'rare':

            if type_seq == 'rare':
                seq_id = get_permuation_rank(
                    sequence, base) - count_likely_before(sequence, base) - 1
                seq_id = seq_id + int(math.factorial(base) / 2)
            else:
                seq_id = count_likely_before(sequence, base)

        else:
            seq_id = get_sequence_rank(sequence, base)
            seq_id += math.factorial(
                base)  # add all permutation at the beginnig
            # remove the already accounted for permutation.
            seq_id = seq_id - count_permutation_before(sequence, base)
    elif dataset_type == 'pair':
        seq_id = brute_force_order(sequence, base, dataset_type)
    return seq_id


def convert_key_sequence_to_int(power_base, histo_dict, fun_key, dataset_type):
    converted_dict = {}
    zero_space = 0
    for key, val in histo_dict.items():
        try:
            tokens = key.replace('[ ', '').replace(']',
                                                   '').replace('[',
                                                               '').split(' ')
            token = []
            for x in tokens:
                try:
                    x_int = int(x)
                    if x_int not in range(power_base):
                        x_int = random.randint(power_base)

                    token.append(x_int)
                except:
                    pass
        except:
            print('problem with', key)
        try:
            ind = fun_key(token, power_base, dataset_type)
            if ind in converted_dict:  # this can happen by random chance but highly unlikely
                converted_dict[ind] += val
            else:
                converted_dict[ind] = val
        except:
            print('problem with', token)
            ind = fun_key(token, power_base, dataset_type)

    return converted_dict