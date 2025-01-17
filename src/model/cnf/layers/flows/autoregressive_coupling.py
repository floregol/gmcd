import torch
import torch.nn as nn
import numpy as np 
import os
import sys
import math
sys.path.append("../../")
from src.mutils import get_device, create_channel_mask
from src.model.cnf.layers.flows.flow_layer import FlowLayer
from src.model.cnf.layers.flows.mixture_cdf_layer import MixtureCDFCoupling


class AutoregressiveMixtureCDFCoupling(FlowLayer):

	def __init__(self, c_in, model_func, block_type=None, num_mixtures=10):
		super().__init__()
		self.c_in = c_in
		self.num_mixtures = num_mixtures
		self.block_type = block_type
		self.scaling_factor = nn.Parameter(torch.zeros(self.c_in))
		self.mixture_scaling_factor = nn.Parameter(torch.zeros(self.c_in, self.num_mixtures))
		self.nn = model_func(c_out=c_in*(2 + 3 * self.num_mixtures))


	def forward(self, z, ldj=None, reverse=False, **kwargs):
		if ldj is None:
			ldj = z.new_zeros(z.size(0),)
		
		if not reverse:
			nn_out = self.nn(x=z, **kwargs)

			t, log_s, log_pi, mixt_t, mixt_log_s = MixtureCDFCoupling.get_mixt_params(nn_out, mask=None,
																				 num_mixtures=self.num_mixtures,
																				 scaling_factor=self.scaling_factor,
																				 mixture_scaling_factor=self.mixture_scaling_factor)
			
			z = z.double()
			z_out, ldj_mixt = MixtureCDFCoupling.run_with_params(z, t, log_s, log_pi, mixt_t, mixt_log_s, reverse=reverse)
		else:
			raise NotImplementedError

		ldj = ldj + ldj_mixt.float()	
		z_out = z_out.float()
		if "channel_padding_mask" in kwargs and kwargs["channel_padding_mask"] is not None:
			z_out = z_out * kwargs["channel_padding_mask"]

		return z_out, ldj


	def info(self):
		s = "Autoregressive Mixture CDF Coupling Layer - Input size %i" % (self.c_in)
		if self.block_type is not None:
			s += ", block type %s" % (self.block_type)
		return s


if __name__ == "__main__":
	torch.manual_seed(42)
	np.random.seed(42)

	batch_size, seq_len, c_in = 1, 3, 3
	hidden_size = 8
	_inp = torch.randn(batch_size, seq_len, c_in)
	lengths = torch.LongTensor([seq_len]*batch_size)
	channel_padding_mask = create_channel_mask(length=lengths, max_len=seq_len)
	time_embed = nn.Linear(2*seq_len, 2)

	module = AutoregressiveMixtureCDFCoupling1D(c_in=c_in, hidden_size=hidden_size, num_mixtures=4, 
												time_embed=time_embed, autoreg_hidden=True)

	orig_out, _ = module(z=_inp, length=lengths, channel_padding_mask=channel_padding_mask)
	print("Out", orig_out)

	_inp[0,1,1] = 10
	alt_out, _ = module(z=_inp, length=lengths, channel_padding_mask=channel_padding_mask)
	print("Out diff", (orig_out - alt_out).abs())