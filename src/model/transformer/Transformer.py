from flash_pytorch import FLASHTransformer
from src.model.transformer.autoregressiveWrapper import AutoregressiveWrapper
import torch
import torch.nn as nn
from src.mutils import get_device


class Transformer(nn.Module):
    def __init__(self,  run_config, dataset_class, name="GMCD", figure_path=""):
        super().__init__()
        self.figure_path = figure_path
        self.name = name
        self.run_config = run_config
        self.set_config()
        self.dataset_class = dataset_class
        self.S = self.run_config.S
        self.K = self.run_config.K
        transformer_model = FLASHTransformer(
            num_tokens=self.S,
            dim=self.run_config.dim,
            depth=self.run_config.depth,
            causal=True,
            group_size=self.K,
            shift_tokens=True
        )

        self.model = AutoregressiveWrapper(transformer_model)
    def set_config(self):
        self.run_config.dim = 64
        self.run_config.depth = 3
    def sample(self, num_samples, **kwargs):
        out_sample = {}
        inp = torch.zeros(num_samples, 1).long().to(get_device())
        x = self.model.generate(inp, self.S, filter_thres=0)

        out_sample['x'] = x
        return out_sample

    def forward(self, z, **kwargs):
        # append class zero at the begining?
        ar_z = torch.zeros(z.shape[0], z.shape[1]+1)
        ar_z[:, 1:] = z
        log_p = -self.model(ar_z.long().to(get_device()))
        return log_p

    def need_data_init(self):
        return False
