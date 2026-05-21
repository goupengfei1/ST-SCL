import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn
from ..utils import check_nan_inf



def l1_loss(input_data, target_data, **kwargs):
    """unmasked mae."""

    return F.l1_loss(input_data, target_data)


def l2_loss(input_data, target_data, **kwargs):
    """unmasked mse"""

    check_nan_inf(input_data)
    check_nan_inf(target_data)
    return F.mse_loss(input_data, target_data)


def masked_mase(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    return 0.3 * masked_rmse(preds=preds, labels=labels, null_val=null_val) + 0.7 * masked_mae(preds=preds, labels=labels, null_val=null_val)

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return  torch.mean(loss) #+  0.5 * masked_mae(preds, labels)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.
                                    Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = torch.where(torch.abs(labels) < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def get_pearson_soft_labels(seq):
    if seq.dim() == 4 and seq.shape[1] == 1:
        seq = seq.squeeze(1)

    B, N, L = seq.shape

    seq_mean = seq.mean(dim=-1, keepdim=True)
    seq_centered = seq - seq_mean

    seq_norm = F.normalize(seq_centered, p=2, dim=-1)
    pearson_matrix = torch.matmul(seq_norm, seq_norm.transpose(1, 2))  # [B, N, N]

    sim_min = pearson_matrix.min(dim=-1, keepdim=True)[0]
    sim_max = pearson_matrix.max(dim=-1, keepdim=True)[0]
    pearson_scaled = (pearson_matrix - sim_min) / (sim_max - sim_min + 1e-8)

    soft_labels = pearson_scaled - 0.5

    soft_labels = torch.sigmoid(soft_labels)
    mask_diag = torch.eye(N, device=seq.device).bool()
    soft_labels.masked_fill_(mask_diag.unsqueeze(0), 0.0)

    return soft_labels

class SpatioTemporalPriorConLoss(nn.Module):
    def __init__(self, node_pos, tau_sim=0.7, tau_spatial=1.0, alpha=0.5):
        super().__init__()
        self.tau_sim = tau_sim
        self.tau_spatial = tau_spatial
        self.node_pos = node_pos.to("cuda:0")
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, features, raw_seq):
        features = features.squeeze(1)

        B, N, D = features.shape


        dist_matrix = torch.cdist(self.node_pos, self.node_pos, p=2)
        spatial_weights = 2.0 * torch.sigmoid(-self.tau_spatial * dist_matrix).to(features.device)
        spatial_weights = spatial_weights.unsqueeze(0).expand(B, -1, -1)

        with torch.no_grad():
            temp_weights = get_pearson_soft_labels(raw_seq)

        final_soft_target = self.alpha * spatial_weights + (1 - self.alpha) * temp_weights
        mask_diag = torch.eye(N, device=features.device).bool()
        final_soft_target.masked_fill_(mask_diag.unsqueeze(0), 0.0)

        final_soft_target = final_soft_target / (final_soft_target.sum(dim=-1, keepdim=True) + 1e-8)

        z = F.normalize(features, p=2, dim=-1)
        sim_matrix = torch.matmul(z, z.transpose(1, 2)) / self.tau_sim

        log_probs = F.log_softmax(sim_matrix, dim=-1)
        loss = -torch.sum(final_soft_target * log_probs, dim=-1)

        return loss.mean()
