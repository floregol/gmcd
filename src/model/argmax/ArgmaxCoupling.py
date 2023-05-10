import torch.nn as nn
from .coupling.flow import ArgmaxCouplingFlow





class ArgmaxCoupling(nn.Module):
    def __init__(self,  run_config, dataset_class, name="ArgmaxCoupling", figure_path=""):
        super().__init__()
        self.figure_path = figure_path
        self.flow_layers = nn.ModuleList()
        self.latent_z_layers = nn.ModuleList()
        self.name = name
        self.run_config = run_config
        self.set_config()
        self.dataset_class = dataset_class
        self.S = self.run_config.S
        self.K = self.run_config.K
        # data_shape, num_classes, run_config
        self.model = ArgmaxCouplingFlow(data_shape=self.S, num_classes=self.K,
                                            num_steps=run_config.num_steps,
                                            actnorm=run_config.actnorm,
                                            num_mixtures=run_config.num_mixtures,
                                            perm_channel=run_config.perm_channel,
                                            perm_length=run_config.perm_length,
                                            base_dist=run_config.base_dist,
                                            encoder_steps=run_config.encoder_steps,
                                            encoder_bins=run_config.encoder_bins,
                                            encoder_ff_size=run_config.encoder_ff_size,
                                            context_size=run_config.context_size,
                                            context_ff_layers=run_config.context_ff_layers,
                                            context_ff_size=run_config.context_ff_size,
                                            context_dropout=run_config.context_dropout,
                                            lstm_layers=run_config.lstm_layers,
                                            lstm_size=run_config.lstm_size,
                                            lstm_dropout=run_config.lstm_dropout,
                                            input_dp_rate=run_config.input_dp_rate)
    def set_config(self):
        self.run_config.context_size = 64
        self.run_config.context_lstm_layers = 1
        self.run_config.context_lstm_size = 32
        self.run_config.encoder_steps = 2
        self.run_config.lstm_size = 64
        self.run_config.lstm_layers = 1
        self.run_config.lstm_dropout = 0.0
        self.run_config.input_dp_rate = 0.0
        self.run_config.num_steps = 2
        if self.run_config.S == 10:
            self.run_config.lstm_size = 128
        
        self.run_config.type = 'coupling'
        self.run_config.num_steps = 2
        self.run_config.num_mixtures = 8
        self.run_config.encoder_bins = None
        self.run_config.encoder_ff_size = 64
        self.run_config.context_ff_layers = 1
        self.run_config.context_ff_size = 64
        self.run_config.context_dropout = 0.0

       
        self.run_config.perm_channel = 'none'
        self.run_config.perm_length = 'reverse'
        self.run_config.base_dist = 'conv_gauss'
        self.run_config.actnorm = False
        self.run_config.encoder_steps = 1
        self.run_config.context_size = 32
        self.run_config.lstm_layers = 1
        self.run_config.lstm_size = 64
        self.run_config.lstm_dropout = 0.0
        self.run_config.context_lstm_layers = 1
        self.run_config.context_lstm_size = 32
        self.run_config.input_dp_rate = 0.0
         
    


    def sample(self, num_samples, **kwargs):
        out_sample = {}
        x = self.model.sample(
            num_samples=num_samples)  # returns shape [b, 1, S]
        out_sample['x'] = x.reshape(num_samples, x.shape[2])
        return out_sample

    def forward(self, z, **kwargs):
        z = z.reshape(z.shape[0], 1, z.shape[1])
        log_p = self.model.log_prob(z)
        return log_p

    def need_data_init(self):
        return False
