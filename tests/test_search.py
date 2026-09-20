import torch

from tinyq.quant.core import (
    dequantize_groupwise,
    quantization_error,
    quantize_groupwise,
    quantize_tensor,
)
from tinyq.quant.search import quantize_groupwise_searched, search_group_params


def _err(w, q, s, z, group):
    return quantization_error(w, dequantize_groupwise(q, s, z, group))["rel_fro"]


def test_search_beats_minmax_with_outliers():
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    w[:, ::37] *= 25  # unos pocos pesos atipicos por grupo

    base = _err(w, *quantize_groupwise(w, 4, 64), 64)
    searched = _err(w, *quantize_groupwise_searched(w, 4, 64), 64)
    assert searched < base


def test_search_never_much_worse_on_clean_weights():
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    base = _err(w, *quantize_groupwise(w, 4, 64), 64)
    searched = _err(w, *quantize_groupwise_searched(w, 4, 64), 64)
    assert searched <= base * 1.05


def test_search_group_params_shapes():
    block = torch.randn(10, 64)
    scale, zero = search_group_params(block, bits=4)
    assert scale.shape == (10, 1)
    assert zero.shape == (10, 1)
    assert (scale > 0).all()


def test_searched_values_stay_in_range():
    torch.manual_seed(0)
    w = torch.randn(8, 128) * 10
    q, s, z = quantize_groupwise_searched(w, bits=4, group_size=32)
    assert q.max() <= 15 and q.min() >= 0
    assert s.dtype == torch.float16 and z.dtype == torch.uint8


def test_quantize_tensor_honors_search_flag():
    """La bandera debe cambiar el resultado: si no, la rama RTN la ignora."""
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    w[:, ::31] *= 20
    plain = quantize_tensor(w, 4, 64, search=False)
    searched = quantize_tensor(w, 4, 64, search=True)
    assert not torch.equal(plain.scales, searched.scales)
    e_plain = quantization_error(w, plain.dequantize())["rel_fro"]
    e_search = quantization_error(w, searched.dequantize())["rel_fro"]
    assert e_search < e_plain


def test_rtn_pipeline_uses_search_when_enabled():
    from tinyq.quantizer import QuantConfig, quantize_model
    from tests.test_pipeline import calib, tiny_llama

    out = {}
    for flag in (False, True):
        model = tiny_llama()
        quantize_model(
            model, calib(n=2),
            QuantConfig(method="rtn", bits=4, group_size=32, search_scale=flag),
        )
        out[flag] = model.model.layers[0].mlp.down_proj.scales.clone()
    assert not torch.equal(out[False], out[True])


def test_symmetric_search_runs():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    q, s, z = quantize_groupwise_searched(w, bits=4, group_size=32, symmetric=True)
    assert q.shape == w.shape
