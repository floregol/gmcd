import torch
import torch.nn as nn
from general.mutils import create_transformer_mask, create_channel_mask
from layers.categorical_encoding.linear_encoding import LinearCategoricalEncoding


class baseCNF(nn.Module):
    def __init__(self, layers=None, name="Flow model", figure_path=""):
        super().__init__()
        self.figure_path = figure_path
        self.flow_layers = nn.ModuleList()
        self.latent_z_layers = nn.ModuleList()
        self.name = name

        if layers is not None:
            self.add_layers(layers)

    def add_layers(self, layers):
        for l in layers:
            self.flow_layers.append(l)
        self.print_overview()

    def _create_layers(self):

        self.encoding_dim = self.run_config.encoding_dim
        self.encoding_layer = LinearCategoricalEncoding(self.run_config, dataset_class=self.dataset_class,
                                                        K=self.K, silence=self.silence)

        self.layers = self._create_cnf_latent_z_layers(
            self.run_config.cnfparams)

        self.flow_layers = nn.ModuleList([self.encoding_layer] +
                                         self.layers)

        for l in self.layers:
            self.latent_z_layers.append(l)

    def sample(self, num_samples, z_in=None, length=None, watch_z_t=False, additional=False):
        out_sample = {}

        kwargs = {}
        kwargs["src_key_padding_mask"] = create_transformer_mask(length)
        kwargs["channel_padding_mask"] = create_channel_mask(length)
        out_sample['x'], _ = self.forward(z_in,
                                          ldj=None,
                                          reverse=True,
                                          length=length,
                                          **kwargs)

        return out_sample

    1

    def forward(self, z, ldj=None, reverse=False, length=None, get_ldj_per_layer=False, **kwargs):

        if length is not None:
            kwargs["src_key_padding_mask"] = create_transformer_mask(
                length)
            kwargs["channel_padding_mask"] = create_channel_mask(length)
            if 'length' not in kwargs:
                kwargs['length'] = length
        if ldj is None:
            ldj = z.new_zeros(z.size(0), dtype=torch.float32)

            ldj_per_layer = []
            x_cat = z
            for layer_index, layer in (enumerate(self.flow_layers)
                                       if not reverse else reversed(
                    list(enumerate(self.flow_layers)))):

                layer_res = layer(z,
                                  reverse=reverse,
                                  get_ldj_per_layer=get_ldj_per_layer, x_cat=x_cat,
                                  **kwargs)

                if len(layer_res) == 2:
                    z, layer_ldj = layer_res
                    detailed_layer_ldj = layer_ldj
                elif len(layer_res) == 3:
                    z, layer_ldj, detailed_layer_ldj = layer_res
                else:
                    print("[!] ERROR: Got more return values than expected: %i" %
                          (len(layer_res)))

                assert torch.isnan(z).sum(
                ) == 0, "[!] ERROR: Found NaN latent values. Layer (%i):\n%s" % (
                    layer_index + 1, layer.info())

                ldj = ldj + layer_ldj
                if isinstance(detailed_layer_ldj, list):
                    ldj_per_layer += detailed_layer_ldj
                else:
                    ldj_per_layer.append(detailed_layer_ldj)

            if get_ldj_per_layer:
                return z, ldj, ldj_per_layer
            else:
                return z, ldj

    def need_data_init(self):
        return any([flow.need_data_init() for flow in self.flow_layers])

    def initialize_data_dependent(self, batch_list):
        # Batch list needs to consist of tuples: (z, kwargs)
        with torch.no_grad():
            for layer_index, layer in enumerate(self.flow_layers):
                print("Processing layer %i..." % (layer_index+1), end="\r")
                batch_list = self.run_data_init_layer(batch_list, layer)

    def run_data_init_layer(self, batch_list, layer):
        if layer.need_data_init():
            stacked_kwargs = {key: [b[1][key] for b in batch_list]
                              for key in batch_list[0][1].keys()}
            for key in stacked_kwargs.keys():
                if isinstance(stacked_kwargs[key][0], torch.Tensor):
                    stacked_kwargs[key] = torch.cat(stacked_kwargs[key], dim=0)
                else:
                    stacked_kwargs[key] = stacked_kwargs[key][0]
            if not (isinstance(batch_list[0][0], tuple) or isinstance(batch_list[0][0], list)):
                input_data = torch.cat([z for z, _ in batch_list], dim=0)
                layer.data_init_forward(input_data, **stacked_kwargs)
            else:
                input_data = [torch.cat([z[i] for z, _ in batch_list], dim=0)
                              for i in range(len(batch_list[0][0]))]
                layer.data_init_forward(*input_data, **stacked_kwargs)
        out_list = []
        for z, kwargs in batch_list:
            if isinstance(z, tuple) or isinstance(z, list):
                z = layer(*z, reverse=False, **kwargs)
                out_list.append([e.detach()
                                for e in z[:-1] if isinstance(e, torch.Tensor)])
                if len(z) == 4 and isinstance(z[-1], dict):
                    kwargs.update(z[-1])
                    out_list[-1] = out_list[-1][:-1]
            else:
                z = layer(z, reverse=False, **kwargs)[0]
                out_list.append(z.detach())
        batch_list = [(out_list[i], batch_list[i][1])
                      for i in range(len(batch_list))]
        return batch_list
