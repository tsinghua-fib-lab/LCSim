import torch
from torch_geometric.data import Batch, HeteroData

KERNEL_NUM = 5
KERNEL_MUL = 2.0


def compute_mmd_vel(
    data: HeteroData, traj: torch.Tensor, mask: torch.Tensor, time_step: int = 11
) -> torch.Tensor:
    """compute the MMD between generated and ground truth velocity

    Args:
        data (HeteroData): batch data
        traj (torch.Tensor): predicted trajectory of shape (num_agents, num_steps, 5).
            The last dimension represents [x, y, length, width, yaw].
        mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

    Returns:
        torch.Tensor: MMD of shape (1, )
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 3
    assert traj.shape[0] == data["agent"]["xyz"].shape[0]

    # only compute vehicles
    vehicle_mask = data["agent"]["type"] == 1
    mask = mask & vehicle_mask
    batch_a = (
        data["agent"]["batch"] if isinstance(data, Batch) else torch.zeros_like(mask)
    )
    batch_size = batch_a.max().item() + 1

    mmd_vel = 0
    for b in range(batch_size):
        mask_b = batch_a == b
        if not torch.any(mask_b & mask):
            continue
        traj_b = traj[mask_b]
        mask_gen = mask[mask_b]
        gen_xy = traj_b[..., :2]
        gen_vel = (
            (gen_xy[:, 1:] - gen_xy[:, :-1])[mask_gen].norm(dim=-1, p=2).reshape(-1, 1)
        )
        real_xy = data["agent"]["xyz"][mask_b][:, time_step:, :2]
        real_vel = (real_xy[:, 1:] - real_xy[:, :-1]).norm(dim=-1, p=2).reshape(-1, 1)
        mmd_vel += mmd(gen_vel, real_vel)
    mmd_vel /= batch_size
    return mmd_vel


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=KERNEL_MUL, kernel_num=KERNEL_NUM, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)
    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)
    return (XX + XY).sum() + (YX + YY).sum()
