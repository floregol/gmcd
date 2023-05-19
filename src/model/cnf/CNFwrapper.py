import torch
import torch.nn as nn
from diffusion_utils.diffusion_multinomial import index_to_log_onehot
from src.mutils import create_channel_mask, create_transformer_mask
from src.model.cnf.CNF import FlowSetModeling
import torch
from src.model.gmcd.distributions import LogisticDistribution
from src.task_template import TaskTemplate


class CNF(nn.Module):
    def __init__(self, run_config, dataset_class, figure_path):
        super().__init__()

        self.name = 'CNF'
        self.run_config = run_config
        self.dataset_class = dataset_class
        self.figure_path = figure_path
        self.S = self.run_config.S
        self.K = self.run_config.K
        self.cnf = FlowSetModeling(self.run_config, dataset_class)
        self.prior_distribution = LogisticDistribution()

    def _preprocess_batch(self, batch):
        x_in = batch
        x_length = x_in.new_zeros(x_in.size(0),
                                  dtype=torch.long) + x_in.size(1)
        x_channel_mask = x_in.new_ones(x_in.size(0),
                                       x_in.size(1),
                                       1,
                                       dtype=torch.float32)
        return x_in, x_length, x_channel_mask

    def sample(self, num_samples, **kwargs):
        out_sample = {}
        z_in = None
        z_length = None
        hidden_size = self.run_config.categ_encoding_num_dimensions

        z_in = self.prior_distribution.sample(shape=(num_samples, self.S,
                                                     hidden_size))
        batch = self._preprocess_batch(z_in)

        z_in, z_length, _ = TaskTemplate.batch_to_device(batch)

        kwargs = {}
        kwargs["src_key_padding_mask"] = create_transformer_mask(z_length)
        kwargs["channel_padding_mask"] = create_channel_mask(z_length)
        out_sample = {}
        out_sample['x'], _ = self.cnf(z_in,
                              ldj=None,
                              reverse=True,
                              length=z_length,
                              **kwargs)

        return out_sample

    def forward(self, z, **kwargs):
        

        if self.training:
            z, ldj, _ = self.cnf(z, **kwargs)
            neglog_prob = -(self.prior_distribution.log_prob(z) *
                            kwargs['x_channel_mask']).sum(dim=[1, 2])
            neg_ldj = -ldj
            return -(neg_ldj + neglog_prob)
        else:
            z, ldj = self.cnf(z, **kwargs)
            neglog_prob = -(self.prior_distribution.log_prob(z) *
                            kwargs['x_channel_mask']).sum(dim=[1, 2])
            neg_ldj = -ldj
            return -(neg_ldj + neglog_prob)

    def need_data_init(self):
        return True

    def initialize_data_dependent(self, batch_list):
        self.cnf.initialize_data_dependent(batch_list)