import numpy as np


class OmegaDelta():
    def __init__(self, seq_to_native_index, native_index_to_seq, Delta,
                 Omega_Delta_dict_native) -> None:
        self.seq_to_native_index = seq_to_native_index
        self.native_index_to_seq = native_index_to_seq
        self.Delta = Delta
        self.Omega_Delta_dict_native = Omega_Delta_dict_native
        self.compute_and_set_val()

    def compute_and_set_val(self):
        self.num_s = len(self.Omega_Delta_dict_native) + 1
        self.flatten_pmf = [
            np.sum([p for i, p in val])
            for key, val in self.Omega_Delta_dict_native.items()
        ]

        start_interval = 0
        self.ground_truth_p = {}
        for key, val in self.Omega_Delta_dict_native.items():
            p = np.mean([p for i, p in val])
            length_interval = len(val)
            self.ground_truth_p[key] = {
                'interval': [start_interval, start_interval + length_interval],
                'p': p
            }
            start_interval += length_interval
        self.list_p_tv = [
            np.sum([
                0.5 * np.abs(p - self.flatten_pmf[key] / len(val))
                for _, p in val
            ]) for key, val in self.Omega_Delta_dict_native.items()
        ]
        self.p_tv_total = np.sum(self.list_p_tv)

    def seq_to_native(self, x):
        tuple_x = tuple(x)
        if tuple_x in self.seq_to_native_index:
            return self.seq_to_native_index[tuple_x]
        else:
            new_native_ind = len(self.seq_to_native_index)
            self.seq_to_native_index[tuple_x] = new_native_ind
            self.native_index_to_seq[new_native_ind] = tuple_x
            return new_native_ind
        

def preprocess_pmf_p(ground_truth_p, num_s):
    # step one sort omf

    p = list(ground_truth_p.values())
    xs = list(ground_truth_p.keys())
    sorting_indices = np.argsort(p)
    p_sorted = np.array(p)[sorting_indices]
    x_sorted = np.array(xs)[sorting_indices]

    p_min = p_sorted[0]
    p_max = p_sorted[-1]
    delta = (p_max - p_min) / (num_s - 1)
    assert delta > 0
    # generate bins
    Omega_Delta_dict_native = {}
    seq_to_native_index = {}
    native_index_to_seq = {}
    for i, x in enumerate(x_sorted):
        seq_to_native_index[tuple(x)] = i
        native_index_to_seq[i] = tuple(x)

        parti_index = int(p_sorted[i] / delta) - 1
        if parti_index not in Omega_Delta_dict_native:
            Omega_Delta_dict_native[parti_index] = [(i, p_sorted[i])]
        else:
            Omega_Delta_dict_native[parti_index].append((i, p_sorted[i]))

    omegaDelta = OmegaDelta(seq_to_native_index, native_index_to_seq, delta,
                            Omega_Delta_dict_native)
    print('we are loosing {0} by cutting in {1}'.format(
        omegaDelta.p_tv_total, num_s))

    return omegaDelta


def convert_to_pattern(omegaDelta, list_of_samples):
    ground_truth_p = omegaDelta.ground_truth_p
    new_list_of_samples = []
    for samples in list_of_samples:
        new_trials = []
        for trial in samples:
            histo_samples = {}
            freq = 1/trial.shape[0]
            for x in trial:
                native_index = omegaDelta.seq_to_native(x)

                if native_index in histo_samples:
                    histo_samples[native_index] += freq
                else:
                    histo_samples[native_index] = freq

            new_trials.append(histo_samples)
        new_list_of_samples.append(new_trials)
    return new_list_of_samples, ground_truth_p
