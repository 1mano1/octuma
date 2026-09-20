import torch

from tinyq.quant.core import quantization_error, quantize_tensor
from tinyq.quant.gptq import GPTQConfig, LayerStats, gptq_quantize
from tinyq.quant.qlinear import QuantLinear


def _fake_layer(in_features=128, out_features=64, seed=0):
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    # activaciones con direcciones dominantes: ahi es donde GPTQ ayuda
    basis = torch.randn(in_features, in_features)
    scale = torch.linspace(4.0, 0.05, in_features).unsqueeze(0)
    x = (torch.randn(512, in_features) * scale) @ basis
    return linear, x


def test_gptq_beats_rtn_on_output_error():
    linear, x = _fake_layer()
    stats = LayerStats(in_features=linear.in_features)
    stats.add_batch(x)

    cfg = GPTQConfig(bits=4, group_size=64)
    qt_gptq, info = gptq_quantize(linear.weight.data, stats.H, cfg)
    qt_rtn = quantize_tensor(linear.weight.data, bits=4, group_size=64)

    ref = x @ linear.weight.data.T
    err_gptq = (x @ qt_gptq.dequantize().T - ref).norm()
    err_rtn = (x @ qt_rtn.dequantize().T - ref).norm()

    assert err_gptq < err_rtn
    assert info["rel_fro"] < 0.2


def test_layer_stats_shape_and_symmetry():
    stats = LayerStats(in_features=32)
    stats.add_batch(torch.randn(10, 4, 32))
    assert stats.H.shape == (32, 32)
    assert torch.allclose(stats.H, stats.H.T, atol=1e-5)
    assert stats.n_samples == 40


def test_quantlinear_matches_dequantized_matmul():
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32)
    q = QuantLinear.from_linear(linear, bits=4, group_size=32)
    x = torch.randn(8, 64)
    expected = torch.nn.functional.linear(x, q.dequantized_weight(), q.bias)
    assert torch.allclose(q(x), expected, atol=1e-5)
    assert q.memory_bytes() < linear.weight.numel() * 2


def test_quantlinear_keeps_outputs_close_to_fp():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 128, bias=False)
    q = QuantLinear.from_linear(linear, bits=4, group_size=64)
    x = torch.randn(16, 256)
    rel = (q(x) - linear(x)).norm() / linear(x).norm()
    assert rel < 0.1
