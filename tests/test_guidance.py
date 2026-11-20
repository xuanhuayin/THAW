import pytest
import torch

from thaw import add_temporal_guidance


@pytest.mark.parametrize("causal", [False, True])
def test_norm_budget_and_projection(causal):
    generator = torch.Generator().manual_seed(7)
    baseline = torch.randn(2, 3, 5, 4, 4, generator=generator)
    conditional = torch.randn(baseline.shape, generator=generator)
    stock = baseline + 5 * (conditional - baseline)
    result = add_temporal_guidance(stock, conditional, baseline, strength=2, causal=causal)
    extra = result - stock
    torch.testing.assert_close(
        extra.flatten(1).norm(dim=1), 2 * (conditional - baseline).flatten(1).norm(dim=1)
    )
    region = extra[:, :, 1:] if causal else extra
    torch.testing.assert_close(region.sum(dim=2), torch.zeros_like(extra[:, :, 0]), rtol=0, atol=3e-6)
    if causal:
        torch.testing.assert_close(extra[:, :, 0], torch.zeros_like(extra[:, :, 0]), rtol=0, atol=0)


def test_degenerate_video_in_batch():
    baseline = torch.zeros(2, 1, 4, 1, 1)
    conditional = torch.tensor([1000, 1000, 1000, 1000.25, 1, 0, 2, -2]).reshape_as(baseline)
    stock = 5 * conditional
    result = add_temporal_guidance(stock, conditional, baseline, strength=2, causal=True)
    assert torch.equal(result[0], stock[0])
    torch.testing.assert_close((result[1] - stock[1]).norm(), 2 * conditional[1].norm())


def test_stock_when_inactive_or_no_temporal_residual():
    conditional = torch.randn(2, 1, 2, 1, 1, dtype=torch.bfloat16)
    baseline = torch.zeros_like(conditional)
    stock = baseline + 5 * conditional
    assert add_temporal_guidance(stock, conditional, baseline, strength=0, causal=False) is stock
    assert add_temporal_guidance(stock, conditional, baseline, strength=2, causal=True) is stock
    assert add_temporal_guidance(stock, conditional, baseline, strength=2, causal=False).dtype == stock.dtype
