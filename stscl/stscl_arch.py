import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from basicts.losses.losses import SpatioTemporalPriorConLoss
from .pos_embedding import TemporalEmbedding


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal_features, time_emb_features):
        context = time_emb_features

        attn_output, attn = self.cross_attn(query=temporal_features,
                                         key=context,
                                         value=context)

        fused_features = self.norm(temporal_features + self.dropout(attn_output))
        return fused_features


class GlobalTemporalContextAttention(nn.Module):
    def __init__(self, d_model, num_layers=3, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, temporal_features, time_emb_features):
        x = temporal_features
        context = time_emb_features

        for i, layer in enumerate(self.layers):
            x = layer(temporal_features=x, time_emb_features=context)

        return x

class LocalCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):

        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value)

        fused = self.norm(query + self.dropout(attn_out))

        return fused


class StaticCoordinateCrossAttention(nn.Module):
    def __init__(self, d_model, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_emb = nn.Linear(3, d_model)
        self.val_emb = nn.Linear(1, d_model)

        self.layers = nn.ModuleList([
            LocalCrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, seq, pos):

        B, N, _ = pos.shape
        query = self.pos_emb(pos.reshape(B * N, 1, 3))
        x = query

        for i, layer in enumerate(self.layers):
            if seq.dim() == 4:
                seq = seq.squeeze(1)
            L = seq.shape[-1]
            current_key_value = self.val_emb(seq.reshape(B * N, L, 1))

            x = layer(query=x, key_value=current_key_value)

        fused = x.reshape(B, N, self.d_model)

        return fused


class STSCL(nn.Module):

    r"""STELLA architecture."""

    def __init__(self, **model_args):
        super().__init__()

        # attributes
        self.num_nodes = model_args["num_nodes"]  # number of stations
        self.num_features = model_args["num_features"] # number of variables
        self.input_len = model_args["input_len"] # input sequence length
        self.d_model = model_args["d_model"] # hidden dimension
        self.output_len = model_args["output_len"] # output sequence length
        self.num_layers = model_args["num_layers"] # number of layers
        self.root_path = model_args["root_path"] # root path of the positional information
        self.dropout = model_args["dropout"] # dropout rate
        self.con = model_args["if_con"]


        # input embedding layer
        self.temporal_embedding = nn.Sequential(nn.Linear(self.input_len, self.d_model),
                                             nn.GELU(),
                                             nn.Dropout(0.1),
                                             nn.Linear(self.d_model, self.d_model))

        self.spatial_embedding = nn.Sequential(nn.Linear(self.num_nodes, self.d_model),
                                               nn.GELU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(self.d_model, self.num_nodes))

        node_pos = np.load(os.path.join(self.root_path, "pos_data.npy"), allow_pickle=True)
        scaler = StandardScaler()
        scaler.fit(node_pos)
        node_pos = scaler.transform(node_pos)
        self.node_pos = torch.from_numpy(node_pos).float()

        if self.con:
            self.cl_loss_fn = SpatioTemporalPriorConLoss(self.node_pos)

        self.time_emb = TemporalEmbedding(self.d_model)
        self.time_enc = GlobalTemporalContextAttention(self.d_model, self.num_layers)
        self.spa_enc = StaticCoordinateCrossAttention(self.d_model,  self.num_layers, dropout=self.dropout)
        self.gate_linear = nn.Linear(self.d_model*2, self.d_model)
        # regression
        self.output_layer = nn.Sequential(nn.Linear(self.d_model, self.d_model*4),
                                          nn.GELU(),
                                          nn.Linear(self.d_model*4, self.output_len))


    def forward(self, history_data: torch.Tensor, **kwargs):
        """forward

                :param x: history data with shape [B, L, N, C]:

                Returns:
                    torch.Tensor: prediction with shape [B, F, N, C]

                """
        x = history_data[..., 0: self.num_features]
        x_mark = history_data[:, :, 0, self.num_features:]
        ori_seq = x.clone()

        # # embedding
        x = x.transpose(1, -1)  # [B, C, N, L]
        B, C, N, L = x.shape

        # seq_mean_spa = x.mean(dim=-1, keepdim=True).detach()
        # seq_std_spa = x.std(dim=-1, keepdim=True).detach()
        # x = (x - seq_mean_spa) / (seq_std_spa + 1e-9)

        spatial_features = self.spatial_embedding(F.normalize(x.squeeze(1).permute(0, 2, 1), p=2, dim=-1)).unsqueeze(1).transpose(-2, -1)
        temporal_features = self.temporal_embedding(x.squeeze(1))

        spatial_features = self.spa_enc(spatial_features, self.node_pos.to(x.device).repeat(B, 1, 1)).unsqueeze(1)
        temporal_features = self.time_enc(temporal_features, self.time_emb(x_mark)).unsqueeze(1)

        combined = torch.cat([spatial_features, temporal_features], dim=-1)
        gate = torch.sigmoid(self.gate_linear(combined))

        repres = gate * spatial_features + (1 - gate) * temporal_features
        prediction = self.output_layer(repres).transpose(1, -1)
        # prediction = prediction * (seq_std_spa + 1e-9) + seq_mean_spa

        if self.con:
            cl_loss = self.cl_loss_fn(repres, ori_seq.transpose(1, -1).detach())
            return prediction, cl_loss, repres
        else:
            return prediction, 0, 0