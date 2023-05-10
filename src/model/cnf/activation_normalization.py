from layers.cnf.flow_layer import FlowLayer
import torch
import torch.nn as nn
import sys
from functools import partial
import numpy as np

sys.path.append("../../")


class ActNormFlow(FlowLayer):
    """
    Normalizes the activations over channels
    """
    def __init__(self, c_in, data_init=True):
        super().__init__()
        self.c_in = c_in
        self.data_init = data_init

        self.bias = nn.Parameter(torch.zeros(1, 1, self.c_in))
        self.scales = nn.Parameter(torch.zeros(1, 1, self.c_in))

    def forward(self,
                z,
                ldj=None,
                reverse=False,
                length=None,
                channel_padding_mask=None,
                **kwargs):
        if ldj is None:
            ldj = z.new_zeros(z.size(0), )
        if length is None:
            if channel_padding_mask is None:
                length = z.size(1)
            else:
                length = channel_padding_mask.squeeze(dim=2).sum(dim=1)
        else:
            length = length.float()

        if not reverse:
            z = (z + self.bias) * torch.exp(self.scales)
            ldj += self.scales.sum(dim=[1, 2]) * length
        else:
            z = z * torch.exp(-self.scales) - self.bias
            ldj += (-self.scales.sum(dim=[1, 2])) * length

        if channel_padding_mask is not None:
            z = z * channel_padding_mask

        assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
        assert torch.isnan(
            ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."

        return z, ldj

    def need_data_init(self):
        return self.data_init

    def data_init_forward(self,
                          input_data,
                          channel_padding_mask=None,
                          **kwargs):
        if channel_padding_mask is None:
            channel_padding_mask = input_data.new_ones(input_data.shape)
        mask = channel_padding_mask
        num_exp = mask.sum(dim=[0, 1], keepdims=True)
        masked_input = input_data * mask

        bias_init = -masked_input.sum(dim=[0, 1], keepdims=True) / num_exp
        self.bias.data = bias_init

        var_data = (((input_data + bias_init)**2) * mask).sum(
            dim=[0, 1], keepdims=True) / num_exp
        scaling_init = -0.5 * var_data.log()
        self.scales.data = scaling_init

        out = (masked_input + self.bias) * torch.exp(self.scales)
        out_mean = (out * mask).sum(dim=[0, 1]) / num_exp.squeeze()
        out_var = torch.sqrt(
            (((out - out_mean)**2) * mask).sum(dim=[0, 1]) / num_exp)
        # print("[INFO - ActNorm] New mean", out_mean)
        # print("[INFO - ActNorm] New variance", out_var)

    def info(self):
        return "Activation Normalizing Flow (c_in=%i)" % (self.c_in)



if __name__ == "__main__":
    pass
