from src.run_config import RunConfig


class SyntheticRunConfig(RunConfig):
    def __init__(self, dataset, num_resample, S, model_name='GMCD') -> None:
        super().__init__()
        self.S = S  # Number of elements in the sets.
        self.K = S
        self.num_resample = num_resample
        self.eval_freq = 500
        self.dataset = dataset
        self.model_name = model_name

        if self.K == 6:
            
            
            self.batch_size = 1024
            self.cdm_T = 100
            if self.dataset == 'pair':
                self.T = 150
                self.encoding_dim = 5
                self.max_iterations = 2000
                #self.alpha = 1.1
                #self.corrected_var = 0.5
                self.transformer_dim = 64
            else:
                self.T = 20
                self.encoding_dim = 6
                self.transformer_dim = 64
            
                self.max_iterations = 1000
            
            self.input_dp_rate = 0.2
            self.transformer_heads = 8
            self.transformer_depth = 2
            self.transformer_blocks = 1
            self.transformer_local_heads = 4
            self.transformer_local_size = 64

        elif self.K == 8:
            self.T = 50
            self.diffusion_steps = self.T
            self.batch_size = 1024
            self.encoding_dim = 8
            self.max_iterations = 3000
            self.transformer_dim = 64
            self.input_dp_rate = 0.2
            self.transformer_heads = 8
            self.transformer_depth = 2
            self.transformer_blocks = 1
            self.transformer_local_heads = 4
            self.transformer_local_size = 64

        elif self.K == 10:
            self.T = 10
            self.diffusion_steps = self.T
            self.batch_size = 1024
            self.encoding_dim = 9
            self.max_iterations = 2000
            self.transformer_dim = 128
            self.input_dp_rate = 0.2
            self.transformer_heads = 8
            self.transformer_depth = 2
            self.transformer_blocks = 1
            self.transformer_local_heads = 4
            self.transformer_local_size = 64
            self.alpha = 1.1
            self.corrected_var = 0.5
        super().set_dependent_config()
