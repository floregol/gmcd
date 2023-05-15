
import numpy as np
import torch
import torch.nn.functional as F
class PriorDistribution(torch.nn.Module):

	GAUSSIAN = 0


	def __init__(self, **kwargs):
		super().__init__()
		self.distribution = self._create_distribution(**kwargs)


	def _create_distribution(self, **kwargs):
		raise NotImplementedError


	def forward(self, shape=None):
		return self.sample(shape=shape)


	def sample(self, shape=None):
		if shape is None:
			return self.distribution.sample()
		else:
			return self.distribution.sample(sample_shape=shape)


	def log_prob(self, x):
		logp = self.distribution.log_prob(x)
		assert torch.isnan(logp).sum() == 0, "[!] ERROR: Found NaN values in log-prob of distribution.\n" + \
				"NaN logp: " + str(torch.isnan(logp).sum().item()) + "\n" + \
				"NaN x: " + str(torch.isnan(x).sum().item()) + ", X(abs) max: " + str(x.abs().max())
		return logp


	def prob(self, x):
		return self.log_prob(x).exp()


	def icdf(self, x):
		assert ((x < 0) | (x > 1)).sum() == 0, \
			   "[!] ERROR: Found values outside the range of 0 to 1 as input to the inverse cumulative distribution function."
		return self.distribution.icdf(x)


	def cdf(self, x):
		return self.distribution.cdf(x)


	def info(self):
		raise NotImplementedError


	

class GaussianDistribution(PriorDistribution):


	def __init__(self, mu=0.0, sigma=1.0, **kwargs):
		super().__init__(mu=mu, sigma=sigma, **kwargs)
		self.mu = mu
		self.sigma = sigma


	def _create_distribution(self, mu=0.0, sigma=1.0, **kwargs):
		return torch.distributions.normal.Normal(loc=mu, scale=sigma)


	def info(self):
		return "Gaussian distribution with mu=%f and sigma=%f" % (self.mu, self.sigma)



class LogisticDistribution(PriorDistribution):


	def __init__(self, mu=0.0, sigma=1.0, eps=1e-4, **kwargs):
		sigma = sigma / 1.81 # STD of a logistic distribution is about 1.81 in default settings
		super().__init__(mu=mu, sigma=sigma)
		self.mu = mu
		self.sigma = sigma
		self.log_sigma = np.log(self.sigma)
		self.eps = eps


	def _create_distribution(self, mu=0.0, sigma=1.0, **kwargs):
		return torch.distributions.uniform.Uniform(low=0.0, high=1.0)


	def _safe_log(self, x, eps=1e-22):
		return torch.log(x.clamp(min=1e-22))


	def _shift_x(self, x):
		return LogisticDistribution.shift_x(x, self.mu, self.sigma, self.log_sigma)

	def _unshift_x(self, x):
		return LogisticDistribution.unshift_x(x, self.mu, self.sigma, self.log_sigma)

	@staticmethod
	def shift_x(x, mu, sigma, log_sigma=None): # x from Uni(0,1) to log(u,s) with log(p(x))
		if log_sigma is None:
			log_sigma = sigma.log()
		x = x.double()
		z = -torch.log(x.reciprocal() - 1.) # ( log(x) - log(1-x)) = log(x/(1-x))
		ldj = -torch.log(x) - torch.log(1. - x)
		z, ldj = z.float(), ldj.float()
		z = z * sigma + mu # z = mu + s *( log(x) - log(1-x))
		ldj = ldj - log_sigma
		return z, ldj

	@staticmethod
	def unshift_x(x, mu, sigma, log_sigma=None): # x from log(u,s) to Uni(0,1)
		if log_sigma is None:
			log_sigma = sigma.log()
		x = (x - mu) / sigma
		z = torch.sigmoid(x)
		ldj = F.softplus(x) + F.softplus(-x) + log_sigma
		return z, ldj


	def sample(self, shape=None, return_ldj=False, temp=1.0):
		samples = super().sample(shape=shape) # sample from a UNIFORM(0,1)
		if shape[-1] != 1:
			samples = samples.squeeze(dim=-1)
		samples = ( samples * (1-self.eps) ) + self.eps/2
		if temp == 1.0:
			samples, sample_ldj = self._shift_x(samples)
		else:
			samples, sample_ldj = SigmoidUniformDistribution.shift_x(samples, self.mu, self.sigma*temp, self.log_sigma + np.log(temp))
		if not return_ldj:
			return samples
		else:
			return samples, sample_ldj


	def log_prob(self, x):
		x, ldj = self._unshift_x(x)
		# Distribution logp not needed as it is 0 anyways
		logp = - ldj

		assert torch.isnan(logp).sum() == 0, "[!] ERROR: Found NaN values in log-prob of distribution.\n" + \
				"NaN logp: " + str(torch.isnan(logp).sum().item()) + "\n" + \
				"NaN x: " + str(torch.isnan(x).sum().item()) + ", X(abs) max: " + str(x.abs().max())

		return logp


	def icdf(self, x, return_ldj=False):
		assert ((x < 0) | (x > 1)).sum() == 0, \
			   "[!] ERROR: Found values outside the range of 0 to 1 as input to the inverse cumulative distribution function."
		z, ldj = self._shift_x(x)
		if not return_ldj:
			return z
		else:
			return z, ldj


	def cdf(self, x, return_ldj=False):
		z, ldj = self._unshift_x(x)
		if not return_ldj:
			return z
		else:
			return z, ldj


	def info(self):
		return "Sigmoid Uniform distribution with mu=%.2f and sigma=%.2f" % (self.mu, self.sigma)



