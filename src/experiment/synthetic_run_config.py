from src.run_config import RunConfig


class SyntheticRunConfig(RunConfig):
    def __init__(self, dataset, num_resample=None, S=None, model_name='GMCD') -> None:
        super().__init__()
        if dataset == 'sort' or dataset == 'pair':
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
                    self.T = 100
                    #self.alpha = 1.1
                    #self.corrected_var = 0.5
                    self.encoding_dim = 5
                    self.transformer_dim = 64
                
                    self.max_iterations = 3000
                    if model_name == 'ArgmaxAR':
                        self.max_iterations = 2500
                else:
                    self.T = 10
                    self.alpha = 1.1
                    self.corrected_var = 0.5
                    self.encoding_dim = 6
                    self.transformer_dim = 64

                    self.max_iterations = 1000
                    if model_name == 'ArgmaxAR':
                        self.max_iterations = 4000
                self.input_dp_rate = 0.2
                self.transformer_heads = 8
                self.transformer_depth = 2
                self.transformer_blocks = 1
                self.transformer_local_heads = 4
                self.transformer_local_size = 64
                if model_name == 'CNF':
                    
                    self.categ_encoding_num_dimensions = 4
                    self.categ_encoding_flow_config = {}
                    self.categ_encoding_flow_config["model_func"] = None
                    self.categ_encoding_flow_config["block_type"] = "Transformer"
                    self.categ_encoding = {}
                    self.encoding_dequantization = False
                    # If selected, the encoder distribution is joint over categorical variables.", action="store_true")
                    self.encoding_variational = False

                    # Flow parameters
                    # Number of flows used in the embedding layer.
                    self.encoding_num_flows = 0
                    # Number of hidden layers of flows used in the parallel embedding layer.
                    self.encoding_hidden_layers = 2
                    # Hidden size of flows used in the parallel embedding layer.
                    self.encoding_hidden_size = 64
                    # Number of mixtures used in the coupling layers (if applicable).
                    self.encoding_num_mixtures = 4

                    self.coupling_hidden_size = 128
                    self.encoding_dim = 4
                    
                    self.coupling_hidden_layers = 2
                    self.coupling_num_flows =4
                    self.coupling_mask_ratio = 0.5
                    self.coupling_num_mixtures = 4
                    self.max_iterations = 7000



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

            else:
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
        
        

        elif dataset == 'real' or dataset == 'proxy':
            self.S = 53  # Number of elements in the sets.
            self.K = 21
            self.encoding_dim = 18
            self.eval_freq = 500
            self.dataset = dataset
            self.model_name = model_name
            self.cdm_T = 1000
            self.T = 200
            self.diffusion_steps = self.T
            self.batch_size = 1024
            self.encoding_dim = 9
            self.max_iterations = 1000
            self.transformer_dim = 128
            self.input_dp_rate = 0.2
            self.transformer_heads = 16
            self.transformer_depth = 12
            self.transformer_blocks = 1
            self.transformer_local_heads = 8
            self.transformer_local_size = 128
            self.alpha = 1.1
            self.corrected_var = 0.5
            if dataset == 'real':
                self.max_iterations = 50
            # if model_name == 'CNF':
                    
            #         self.categ_encoding_num_dimensions = 4
            #         self.categ_encoding_flow_config = {}
            #         self.categ_encoding_flow_config["model_func"] = None
            #         self.categ_encoding_flow_config["block_type"] = "Transformer"
            #         self.categ_encoding = {}
            #         self.encoding_dequantization = False
            #         # If selected, the encoder distribution is joint over categorical variables.", action="store_true")
            #         self.encoding_variational = False

            #         # Flow parameters
            #         # Number of flows used in the embedding layer.
            #         self.encoding_num_flows = 0
            #         # Number of hidden layers of flows used in the parallel embedding layer.
            #         self.encoding_hidden_layers = 2
            #         # Hidden size of flows used in the parallel embedding layer.
            #         self.encoding_hidden_size = 64
            #         # Number of mixtures used in the coupling layers (if applicable).
            #         self.encoding_num_mixtures = 4

            #         self.coupling_hidden_size = 128
            #         self.encoding_dim = 4
                    
            #         self.coupling_hidden_layers = 2
            #         self.coupling_num_flows =4
            #         self.coupling_mask_ratio = 0.5
            #         self.coupling_num_mixtures = 4
            #         self.max_iterations = 7000

        super().set_dependent_config()
