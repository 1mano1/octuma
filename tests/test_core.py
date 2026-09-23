import pytest
import torch

from octuma.quant.core import (
    dequantize_groupwise,
    pack_bits,
    quantization_error,
    quantize_groupwise,
    quantize_tensor,
    unpack_bits,
)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_roundtrip(bits):
    torch.manual_seed(0)
    qmax = (1 << bits) - 1
    q = torch.randint(0, qmax + 1, (16, 128), dtype=torch.uint8)
    packed = pack_bits(q, bits)
    assert packed.numel() == q.numel() * bits // 8
    assert torch.equal(unpack_bits(packed, bits, q.shape[1]), q)


@pytest.mark.parametrize("bits,max_rel", [(8, 0.01), (4, 0.12), (2, 0.5)])
def test_error_within_expected_budget(bits, max_rel):
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    qt = quantize_tensor(w, bits=bits, group_size=64)
    err = quantization_error(w, qt.dequantize())
    assert err["rel_fro"] < max_rel


def test_error_shrinks_with_bits():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    errs = [
        quantization_error(w, quantize_tensor(w, bits=b, group_size=64).dequantize())["rel_fro"]
        for b in (2, 4, 8)
    ]
    assert errs[0] > errs[1] > errs[2]


def test_asymmetric_beats_symmetric_on_skewed_weights():
    torch.manual_seed(0)
    w = torch.randn(32, 128).abs() + 0.5  # todos positivos: el cero se desperdicia
    asym = quantization_error(w, quantize_tensor(w, 4, 64, symmetric=False).dequantize())
    sym = quantization_error(w, quantize_tensor(w, 4, 64, symmetric=True).dequantize())
    assert asym["rel_fro"] < sym["rel_fro"]


def test_smaller_groups_reduce_error():
    torch.manual_seed(0)
    w = torch.randn(32, 512)
    w[:, :64] *= 40  # un bloque con valores atipicos
    big = quantization_error(w, quantize_tensor(w, 4, 512).dequantize())["rel_fro"]
    small = quantization_error(w, quantize_tensor(w, 4, 32).dequantize())["rel_fro"]
    assert small < big


def test_memory_savings_int4():
    w = torch.randn(256, 512)
    qt = quantize_tensor(w, bits=4, group_size=64)
    fp16_bytes = w.numel() * 2
    assert qt.nbytes() < fp16_bytes * 0.32  # ~4.5 bits/peso vs 16


def test_group_size_must_divide_in_features():
    with pytest.raises(ValueError):
        quantize_groupwise(torch.randn(4, 100), bits=4, group_size=64)


def test_dequantize_matches_manual_path():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    q, scales, zeros = quantize_groupwise(w, bits=4, group_size=32)
    manual = dequantize_groupwise(q, scales, zeros, 32)
    qt = quantize_tensor(w, bits=4, group_size=32)
    assert torch.allclose(manual, qt.dequantize(), atol=1e-5)
