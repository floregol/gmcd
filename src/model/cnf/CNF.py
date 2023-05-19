import torch
import torch.nn as nn
from src.mutils import get_param_val, create_transformer_mask, create_channel_mask
from src.model.cnf.layers.flows.flow_model import FlowModel
from src.model.cnf.layers.flows.activation_normalization import ActNormFlow
from src.model.cnf.layers.flows.permutation_layers import InvertibleConv
from src.model.cnf.layers.flows.coupling_layer import CouplingLayer
from src.model.cnf.layers.flows.mixture_cdf_layer import MixtureCDFCoupling
from src.model.cnf.layers.categorical_encoding.mutils import create_encoding


class FlowSetModeling(FlowModel):
    def __init__(self, model_params, dataset_class):
        super().__init__(layers=None, name="Set Modeling Flow")
        self.model_params = model_params
        self.dataset_class = dataset_class
        self.set_size = self.model_params.S
        self.vocab_size = self.model_params.K

        self._create_layers()
        self.print_overview()

    def _create_layers(self):

        self.latent_dim = self.model_params.categ_encoding_num_dimensions
        model_func = lambda c_out: CouplingTransformerNet(
            c_in=self.latent_dim,
            c_out=c_out,
            num_layers=self.model_params.coupling_hidden_layers,
            hidden_size=self.model_params.coupling_hidden_size)
        self.model_params.categ_encoding_flow_config["model_func"] = model_func
        self.model_params.categ_encoding_flow_config[
            "block_type"] = "Transformer"
        self.encoding_layer = create_encoding(self.model_params,
                                              dataset_class=self.dataset_class,
                                              vocab_size=self.vocab_size)

        num_flows = self.model_params.coupling_num_flows
        if self.latent_dim > 1:
            coupling_mask = CouplingLayer.create_channel_mask(
                self.latent_dim, ratio=self.model_params.coupling_mask_ratio)
            coupling_mask_func = lambda flow_index: coupling_mask
        else:
            coupling_mask = CouplingLayer.create_chess_mask()
            coupling_mask_func = lambda flow_index: coupling_mask if flow_index % 2 == 0 else 1 - coupling_mask

        layers = []
        for flow_index in range(num_flows):
            layers += [
                ActNormFlow(self.latent_dim),
                InvertibleConv(self.latent_dim),
                MixtureCDFCoupling(
                    c_in=self.latent_dim,
                    mask=coupling_mask_func(flow_index),
                    model_func=model_func,
                    block_type="Transformer",
                    num_mixtures=self.model_params.coupling_num_mixtures)
            ]

        self.flow_layers = nn.ModuleList([self.encoding_layer] + layers)

    def forward(self, z, ldj=None, reverse=False, length=None, **kwargs):
        if length is not None:
            kwargs["src_key_padding_mask"] = create_transformer_mask(length)
            kwargs["channel_padding_mask"] = create_channel_mask(length)
        return super().forward(z,
                               ldj=ldj,
                               reverse=reverse,
                               length=length,
                               **kwargs)

    def get_inner_activations(self,
                              z,
                              length=None,
                              return_names=False,
                              **kwargs):
        if length is not None:
            kwargs["length"] = length
            kwargs["src_key_padding_mask"] = create_transformer_mask(length)
            kwargs["channel_padding_mask"] = create_channel_mask(length)

        out_per_layer = []
        layer_names = []
        for layer_index, layer in enumerate(self.flow_layers):
            z = self._run_layer(layer, z, reverse=False, **kwargs)[0]
            out_per_layer.append(z.detach())
            layer_names.append(layer.__class__.__name__)

        if not return_names:
            return out_per_layer
        else:
            return out_per_layer, layer_names

    def initialize_data_dependent(self, batch_list):
        # Batch list needs to consist of tuples: (z, kwargs)
        print("Initializing data dependent...")
        with torch.no_grad():
            for batch, kwargs in batch_list:
                kwargs["src_key_padding_mask"] = create_transformer_mask(
                    kwargs["length"])
                kwargs["channel_padding_mask"] = create_channel_mask(
                    kwargs["length"])
            for layer_index, layer in enumerate(self.flow_layers):
                batch_list = FlowModel.run_data_init_layer(batch_list, layer)


class CouplingTransformerNet(nn.Module):
    def __init__(self, c_in, c_out, num_layers, hidden_size):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(c_in, hidden_size),
                                         nn.GELU(),
                                         nn.Linear(hidden_size, hidden_size))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size,
                                       nhead=4,
                                       dim_feedforward=2 * hidden_size,
                                       dropout=0.0,
                                       activation='gelu')
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_size),
                                          nn.Linear(hidden_size, hidden_size),
                                          nn.GELU(),
                                          nn.Linear(hidden_size, c_out))

    def forward(self, x, src_key_padding_mask, **kwargs):
        x = x.transpose(
            0, 1
        )  # Transformer layer expects [Sequence length, Batch size, Hidden size]
        x = self.input_layer(x)
        for transformer in self.transformer_layers:
            x = transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.output_layer(x)
        x = x.transpose(0, 1)
        return x
