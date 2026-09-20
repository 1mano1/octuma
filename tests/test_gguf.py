import json

import numpy as np
import pytest
import torch

from tinyq.export.gguf_export import _q4_1_blocks, _q8_0_blocks, export_gguf
from tinyq.quant.core import quantize_groupwise
from tinyq.quantizer import QuantConfig, quantize_model

from .test_pipeline import VOCAB, calib, tiny_llama

gguf = pytest.importorskip("gguf")


def test_q4_1_block_layout_matches_llama_cpp():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    q, scales, zeros = quantize_groupwise(w, bits=4, group_size=32)
    blocks = _q4_1_blocks(q, scales, zeros)

    assert blocks.shape == (4, 2 * 20)  # 2 bloques de 20 bytes por fila

    # reconstruye el primer bloque como lo hace llama.cpp
    raw = blocks[0, :20]
    d = raw[0:2].view(np.float16)[0]
    m = raw[2:4].view(np.float16)[0]
    qs = raw[4:20]
    vals = np.concatenate([qs & 0x0F, qs >> 4]).astype(np.float32)
    recon = torch.tensor(d.astype(np.float32) * vals + m.astype(np.float32))

    expected = (q[0, :32].float() - zeros[0, 0].float()) * scales[0, 0].float()
    assert torch.allclose(recon, expected, atol=2e-3)


def test_q8_0_roundtrip_is_accurate():
    torch.manual_seed(0)
    w = torch.randn(2, 64)
    blocks = _q8_0_blocks(w)
    raw = blocks[0, :34]
    d = raw[0:2].view(np.float16)[0].astype(np.float32)
    q = raw[2:34].view(np.int8).astype(np.float32)
    recon = torch.tensor(d * q)
    rel = (recon - w[0, :32]).norm() / w[0, :32].norm()
    assert rel < 0.02


def _fake_tokenizer_json(path, vocab_size=VOCAB):
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": vocab, "merges": []}, "added_tokens": []}),
        encoding="utf-8",
    )


def test_export_gguf_writes_readable_file(tmp_path):
    model = tiny_llama()
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    _fake_tokenizer_json(tmp_path)

    out = export_gguf(model, tmp_path / "tiny.gguf", tmp_path, name="tiny")
    assert out.stat().st_size > 0

    reader = gguf.GGUFReader(str(out))
    names = {t.name for t in reader.tensors}
    assert "blk.0.attn_q.weight" in names
    assert "token_embd.weight" in names

    q4 = [t for t in reader.tensors if t.name == "blk.0.attn_q.weight"][0]
    assert q4.tensor_type == gguf.GGMLQuantizationType.Q4_1


def test_export_includes_attention_biases(tmp_path):
    """Qwen2 usa sesgo en q, k y v: si no se exportan, el modelo delira."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_bias=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    _fake_tokenizer_json(tmp_path)

    out = export_gguf(model, tmp_path / "bias.gguf", tmp_path)
    names = {t.name for t in gguf.GGUFReader(str(out)).tensors}
    for part in ("attn_q", "attn_k", "attn_v"):
        assert f"blk.0.{part}.weight" in names
        assert f"blk.0.{part}.bias" in names, f"falta el sesgo de {part}"


def test_export_gguf_rejects_wrong_group_size(tmp_path):
    model = tiny_llama()
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=64))
    _fake_tokenizer_json(tmp_path)
    with pytest.raises(ValueError, match="group 32"):
        export_gguf(model, tmp_path / "bad.gguf", tmp_path)
