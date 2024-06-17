import torch
from torch_geometric.data import Batch, HeteroData

from .geometry import compute_pairwise_overlaps


def compute_overlap_rate(
    data: HeteroData, traj: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """compute the overlap rate between agents of predicted trajectory

    Args:
        data (HeteroData): batch data
        traj (torch.Tensor): predicted trajectory of shape (num_agents, num_steps, 5).
            The last dimension represents [x, y, length, width, yaw].
        mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

    Returns:
        torch.Tensor: overlap rate of shape (1, )
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 3
    assert traj.shape[0] == data["agent"]["xyz"].shape[0]

    batch = (
        data["agent"]["batch"] if isinstance(data, Batch) else torch.zeros_like(mask)
    )
    batch_size = batch.max().item() + 1
    # traj = traj[mask]
    overlap_cnt = 0
    for i in range(traj.shape[1]):
        # for b in range(batch_size):
        #     mask_b = batch == b
        #     if not torch.any(mask_b & mask):
        #         continue
        #     traj_b = traj[mask_b, i]
        #     mask_b = mask[mask_b]
        #     overlap = compute_pairwise_overlaps(traj_b)
        #     overlap_cnt += torch.sum(overlap & mask_b)
        overlap = compute_pairwise_overlaps(traj[:, i])
        overlap_cnt += torch.sum(overlap & mask)
    overlap_rate = overlap_cnt / torch.sum(mask) / traj.shape[1]
    return overlap_rate
