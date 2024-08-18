import torch
from torch_geometric.data import Batch, HeteroData

NUM_PREDS = 6


def compute_min_ade(
    data: HeteroData, traj: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """compute the minimum average displacement error (minADE) of predicted trajectory

    Args:
        data (HeteroData): batch data
        traj (torch.Tensor): predicted trajectory of shape (num_agents, num_pred, num_steps, 5).
            The last dimension represents [x, y, length, width, yaw].
        mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

    Returns:
        torch.Tensor: minADE of shape (1, )
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 4
    assert traj.shape[0] == data["agent"]["xyz"].shape[0]

    num_future_steps = traj.shape[2]
    xy = data["agent"]["xyz"][:, -num_future_steps:, :2]
    preds_xy = traj[..., :2]
    min_ade = (
        torch.norm(xy[mask].unsqueeze(1) - preds_xy[mask], dim=-1, p=2)
        .mean(dim=-1)
        .min(dim=-1)[0]
    ).mean()
    return min_ade


def compute_min_fde(
    data: HeteroData, traj: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """compute the minimum final displacement error (minFDE) of predicted trajectory

    Args:
        data (HeteroData): batch data
        traj (torch.Tensor): predicted trajectory of shape (num_agents, num_pred, num_steps, 5).
            The last dimension represents [x, y, length, width, yaw].
        mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

    Returns:
        torch.Tensor: minFDE of shape (1, )
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 4
    assert traj.shape[0] == data["agent"]["xyz"].shape[0]

    num_future_steps = traj.shape[2]
    xy = data["agent"]["xyz"][:, -num_future_steps:, :2]
    preds_xy = traj[..., :2]
    min_fde = (
        torch.norm((xy[mask].unsqueeze(1) - preds_xy[mask])[..., -1, :], dim=-1, p=2)
        .min(dim=-1)[0]
        .mean()
    )
    return min_fde
