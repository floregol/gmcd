import torch
import torch.nn as nn
from diffusion_utils.diffusion_multinomial import MultinomialDiffusion
from diffusion_utils.diffusion_multinomial import index_to_log_onehot
from src.model.cdm.transformer import LinearAttentionTransformerEmbedding
import torch


class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x


class DynamicsTransformer(nn.Module):
    def __init__(self, S, K, run_config):
        super(DynamicsTransformer, self).__init__()
        self.transformer = LinearAttentionTransformerEmbedding(
            input_dim=K,
            output_dim=K,
            dim=run_config.transformer_dim,
            heads=run_config.transformer_heads,
            depth=run_config.transformer_depth,
            n_blocks=run_config.transformer_blocks,
            max_seq_len=S,
            num_timesteps=run_config.diffusion_steps,
            causal=False,  # auto-regressive or not
            ff_dropout=0,  # dropout for feedforward
            # dropout right after self-attention layer
            attn_layer_dropout=run_config.input_dp_rate,
            attn_dropout=0,  # dropout post-attention
            n_local_attn_heads=run_config.transformer_local_heads,
            # number of local attention heads for (qk)v attention.
            # this can be a tuple specifying the exact number of local
            # attention heads at that depth
            # receptive field of the local attention
            local_attn_window_size=run_config.transformer_local_size,
            reversible=False  # use reversible nets, from Reformer paper
        )

        self.rezero = Rezero()

    def forward(self, t, x):
        x = self.transformer(x, t)
        x = x.permute(0, 2, 1)
        x = self.rezero(x)
        return x


class CDM(nn.Module):
    def __init__(self,  run_config, dataset_class, figure_path):
        super().__init__()
        self.flow_layers = nn.ModuleList()
        self.latent_z_layers = nn.ModuleList()
        self.name = 'CDM'
        self.run_config = run_config
        self.dataset_class = dataset_class
        self.figure_path= figure_path
        self.S = self.run_config.S
        self.K = self.run_config.K

        dynamics = DynamicsTransformer(self.S, self.K, run_config)
        self.cdm = MultinomialDiffusion(
            self.K, (self.S,), dynamics,
            timesteps=run_config.diffusion_steps*10,
            loss_type='vb_stochastic',
            parametrization='x0')

    def sample(self, num_samples, **kwargs):
        out_sample = {}
        out_sample['x'] = self.cdm.sample(num_samples=num_samples)

        return out_sample

    def forward(self, z, **kwargs):

        if self.training:
            log_p = self.cdm.log_prob(z)
            return log_p
        else:
            log_x = index_to_log_onehot(z, self.K)
            log_likelihood_other = -self.cdm.nll(log_x)
            return log_likelihood_other

    def need_data_init(self):
        return False