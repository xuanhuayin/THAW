"""THAW's norm-matched temporal residual update."""
import torch


def add_temporal_guidance(
    cfg_prediction: torch.Tensor,
    conditional: torch.Tensor,
    baseline: torch.Tensor,
    *,
    strength: float,
    causal: bool,
) -> torch.Tensor:
    """Apply one active THAW step to tensors of shape [B, C, F, H, W].

    Pass the sampler's original CFG prediction so its arithmetic and dtype
    are preserved. Call this function only inside the selected early window.
    """
    if strength == 0 or conditional.shape[2] <= (2 if causal else 1):
        return cfg_prediction

    difference = conditional.float() - baseline.float()
    if causal:
        tail = difference[:, :, 1:]
        residual = torch.cat(
            (torch.zeros_like(difference[:, :, :1]), tail - tail.mean(dim=2, keepdim=True)),
            dim=2,
        )
    else:
        residual = difference - difference.mean(dim=2, keepdim=True)

    reduce_dims = (1, 2, 3, 4)
    difference_norm = difference.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    residual_norm = residual.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    gain = torch.where(
        residual_norm > 1e-3 * difference_norm,
        strength * difference_norm / (residual_norm + 1e-30),
        torch.zeros_like(residual_norm),
    )
    return (cfg_prediction.float() + gain * residual).to(cfg_prediction.dtype)
