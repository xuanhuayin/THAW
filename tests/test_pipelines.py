from inspect import signature
from types import SimpleNamespace

import pytest
import torch

from thaw import add_temporal_guidance
from thaw.pipelines import use_thaw


class Transformer(torch.nn.Module):
    def forward(self, prediction):
        return (prediction,)


class Scheduler:
    def step(self, model_output, timestep=None, sample=None, *, generator=None, eta=0.0):
        return (model_output,)


@pytest.mark.parametrize("backend", ["wan", "wan_i2v", "cogvideox", "ltx", "hunyuan"])
def test_layout_window_and_native_scheduler_signature(backend):
    pipe = SimpleNamespace(transformer=Transformer(), scheduler=Scheduler())
    shape = (4, 4, 4)
    generator = torch.Generator().manual_seed(39)
    conditional = torch.randn((2, 3, *shape), generator=generator)
    baseline = torch.randn(conditional.shape, generator=generator)
    causal = backend != "cogvideox"

    def identity(x):
        return x

    to_native = identity
    if backend == "cogvideox":
        def to_native(x):
            return x.transpose(1, 2).contiguous()
    if backend == "ltx":
        diffusers = pytest.importorskip("diffusers")
        pipe._pack_latents = diffusers.LTXPipeline._pack_latents
        pipe._unpack_latents = diffusers.LTXPipeline._unpack_latents
        pipe.transformer_spatial_patch_size = 2
        pipe.transformer_temporal_patch_size = 2

        def to_native(x):
            return pipe._pack_latents(x, 2, 2)

    stock = baseline + 5 * (conditional - baseline)
    target = add_temporal_guidance(stock, conditional, baseline, strength=2, causal=causal)
    cond, base = to_native(conditional), to_native(baseline)
    original = pipe.scheduler.step

    with use_thaw(pipe, backend=backend, strength=2, active_steps=2, causal=causal, latent_shape=shape):
        assert signature(pipe.scheduler.step) == signature(original)
        for step in range(4):
            if backend in ("cogvideox", "ltx"):
                pipe.transformer(torch.cat([base, cond]))
            else:
                pipe.transformer(cond)
                pipe.transformer(base)
            actual, = pipe.scheduler.step(to_native(stock), eta=0.0)
            torch.testing.assert_close(actual, to_native(target if step < 2 else stock))
    assert pipe.scheduler.step == original
    assert not pipe.transformer._forward_hooks


def test_failed_call_removes_hooks():
    pipe = SimpleNamespace(transformer=Transformer(), scheduler=Scheduler())
    original = pipe.scheduler.step
    with pytest.raises(RuntimeError, match="Expected two CFG predictions"):
        with use_thaw(pipe, backend="wan", strength=2, active_steps=1, causal=True, latent_shape=(4, 1, 1)):
            pipe.transformer(torch.zeros(1, 1, 4, 1, 1))
            pipe.scheduler.step(torch.zeros(1, 1, 4, 1, 1))
    assert pipe.scheduler.step == original
    assert not pipe.transformer._forward_hooks
