import torch

from tinyq.quant.awq import (
    ScalingGroup,
    apply_scales,
    awq_scale_block,
    llama_like_groups,
    search_scales,
)
from tinyq.quant.core import quantize_tensor

from .test_pipeline import tiny_llama


def _skewed_inputs(in_features=64, n=512, seed=0):
    """Activaciones con unos canales mucho mas grandes que el resto."""
    torch.manual_seed(seed)
    scale = torch.ones(in_features)
    scale[:4] = 30.0
    return torch.randn(n, in_features) * scale


def test_search_scales_finds_useful_alpha():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False)
    x = _skewed_inputs()
    scales, alpha, rel = search_scales([lin], x, bits=4, group_size=32)
    assert rel < 1.0  # mejora respecto a no escalar
    assert 0.0 < alpha <= 1.0
    assert scales.shape == (64,)


def test_apply_scales_preserves_function_with_norm():
    torch.manual_seed(0)
    norm = torch.nn.RMSNorm(64) if hasattr(torch.nn, "RMSNorm") else None
    if norm is None:
        import pytest

        pytest.skip("torch sin RMSNorm")
    lin = torch.nn.Linear(64, 32, bias=False)
    x = torch.randn(8, 64)
    before = lin(norm(x))

    scales = torch.rand(64) + 0.5
    apply_scales(ScalingGroup("g", norm, [lin], True), scales)
    after = lin(norm(x))
    assert torch.allclose(before, after, atol=1e-5)


def test_apply_scales_preserves_function_with_linear_prev():
    torch.manual_seed(0)
    prev = torch.nn.Linear(32, 64)
    lin = torch.nn.Linear(64, 16, bias=False)
    x = torch.randn(8, 32)
    before = lin(prev(x))

    scales = torch.rand(64) + 0.5
    apply_scales(ScalingGroup("g", prev, [lin], False), scales)
    assert torch.allclose(before, lin(prev(x)), atol=1e-4)


def test_awq_reduces_quantization_error_on_skewed_activations():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False)
    x = _skewed_inputs()
    ref = x @ lin.weight.data.T

    plain = quantize_tensor(lin.weight.data, 4, 32).dequantize()
    err_plain = ((x @ plain.T - ref).norm() / ref.norm()).item()

    scales, _, _ = search_scales([lin], x, bits=4, group_size=32)
    w_scaled = lin.weight.data.float() * scales.unsqueeze(0)
    w_hat = quantize_tensor(w_scaled, 4, 32).dequantize() / scales.unsqueeze(0)
    err_awq = ((x @ w_hat.T - ref).norm() / ref.norm()).item()

    assert err_awq < err_plain


def test_groups_detected_on_llama_block():
    model = tiny_llama()
    block = model.model.layers[0]
    groups = llama_like_groups(block, n_heads=4, n_kv_heads=2)
    names = [g.name for g in groups]
    assert "attn_qkv" in names and "mlp_in" in names and "mlp_out" in names
    # con GQA (4 vs 2 cabezas) no se puede plegar o_proj en v_proj
    assert "attn_out" not in names


def test_awq_scale_block_runs_and_reports():
    model = tiny_llama()
    block = model.model.layers[0]
    inputs = {
        block.self_attn.q_proj: torch.randn(64, 64),
        block.mlp.gate_proj: torch.randn(64, 64),
        block.mlp.down_proj: torch.randn(64, 128),
    }
    report, applied = awq_scale_block(
        block, inputs, n_heads=4, n_kv_heads=2, bits=4, group_size=32
    )
    assert {r["group"] for r in report} == {"attn_qkv", "mlp_in", "mlp_out"}
    assert all(0.0 <= r["alpha"] <= 1.0 for r in report)
    for lin, scales in applied.items():
        assert scales.shape == (lin.in_features,)
