import math
import torch
import torch.nn as nn
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer import LinearAttentionTransformer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x


class LinearAttentionTransformerEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dim_emb, dim, depth, n_blocks, max_seq_len, num_timesteps, heads=8, dim_head=None,  reversible=False, ff_chunks=1, ff_glu=False, ff_dropout=0., attn_layer_dropout=0., attn_dropout=0., blindspot_size=1, n_local_attn_heads=0, local_attn_window_size=128, return_embeddings=False, receives_context=False, pkm_layers=tuple(), pkm_num_keys=128, attend_axially=False, linformer_settings=None, context_linformer_settings=None):
        local_attn_window_size = max_seq_len
        assert (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        super().__init__()
        # emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.depth = depth
        emb_dim = dim
        self.emb_dim = emb_dim

        self.depth = depth
        self.n_blocks = n_blocks

        self.first = nn.Linear(dim_emb, emb_dim)

        self.time_pos_emb = SinusoidalPosEmb(emb_dim, num_timesteps)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.Softplus(),
            nn.Linear(emb_dim * 4, emb_dim * n_blocks * depth)
        )

        # self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(
            max_seq_len // local_attn_window_size, local_attn_window_size))

        self.transformer_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.transformer_blocks.append(nn.ModuleList())
            for j in range(depth):
                self.transformer_blocks[-1].append(
                    LinearAttentionTransformer(
                        dim, 1, max_seq_len, heads=heads, dim_head=dim_head,
                        causal=False,
                        ff_chunks=ff_chunks, ff_glu=ff_glu,
                        ff_dropout=ff_dropout,
                        attn_layer_dropout=attn_layer_dropout,
                        attn_dropout=attn_dropout, reversible=reversible,
                        blindspot_size=blindspot_size,
                        n_local_attn_heads=n_local_attn_heads,
                        local_attn_window_size=local_attn_window_size,
                        receives_context=receives_context,
                        pkm_layers=pkm_layers, pkm_num_keys=pkm_num_keys,
                        attend_axially=attend_axially,
                        linformer_settings=linformer_settings,
                        context_linformer_settings=context_linformer_settings))

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(
            emb_dim, output_dim) if not return_embeddings else nn.Identity()

    def forward(self, z, t, **kwargs):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        time_embed = t.view(z.size(0), 1, self.emb_dim,
                            self.n_blocks, self.depth)
        z = self.first(z)
        z_embed_axial = z + self.axial_pos_emb(z).type(z.type())
        # x_embed_axial_time = x_embed_axial + time_embed
        h = torch.zeros_like(z_embed_axial)

        for i, block in enumerate(self.transformer_blocks):
            h = h + z_embed_axial
            for j, transformer in enumerate(block):
                h = transformer(h + time_embed[..., i, j])

        h = self.norm(h)
        return self.out(h)


class DenoisingTransformer(torch.nn.Module):
    def __init__(self, K, S, latent_dim, diffusion_params):
        super(DenoisingTransformer, self).__init__()
        self.transformer = LinearAttentionTransformerEmbedding(
            input_dim=K,
            output_dim=K,
            dim_emb=latent_dim,
            dim=diffusion_params.transformer_dim,
            heads=diffusion_params.transformer_heads,
            depth=diffusion_params.transformer_depth,
            n_blocks=diffusion_params.transformer_blocks,
            max_seq_len=S,
            num_timesteps=diffusion_params.T,
            ff_dropout=0,  # dropout for feedforward
            attn_layer_dropout=diffusion_params.input_dp_rate,
            # dropout right after self-attention layer
            attn_dropout=0,  # dropout post-attention
            n_local_attn_heads=diffusion_params.transformer_local_heads,
            # number of local attention heads for (qk)v attention.
            # this can be a tuple specifying the exact number of local
            # attention heads at that depth
            local_attn_window_size=diffusion_params.transformer_local_size,
            # receptive field of the local attention
            reversible=False,
            # use reversible nets, from Reformer paper
        )

        self.rezero = Rezero()

    def forward(self, t, x):
        x = self.transformer(x, t)
        return x
