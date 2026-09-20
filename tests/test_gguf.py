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


def test_rope_theta_anidado_en_rope_parameters():
    """transformers 5 movio rope_theta dentro de rope_parameters.

    El `getattr(cfg, "rope_theta", 10000.0)` de antes se tragaba el cambio y
    escribia 10000 donde Qwen2.5 usa 1000000. El modelo seguia respondiendo
    frases cortas y la perplejidad se duplicaba en ventanas largas.
    """
    from types import SimpleNamespace

    from tinyq.export.gguf_export import _rope_theta

    plano = SimpleNamespace(rope_theta=1000000.0)
    assert _rope_theta(plano) == 1000000.0

    anidado = SimpleNamespace(rope_parameters={"rope_theta": 1000000.0})
    assert _rope_theta(anidado) == 1000000.0

    # sin el valor NO se inventa un default: eso es lo que corrompia el modelo
    with pytest.raises(ValueError, match="rope_theta"):
        _rope_theta(SimpleNamespace())


def test_tokens_especiales_desde_added_tokens(tmp_path):
    """Los especiales de Qwen (<|im_end|>) viven en added_tokens, no en vocab.

    Buscarlos solo en model.vocab los perdia en silencio y el GGUF salia sin
    eos, asi que llama.cpp no sabia cuando parar de generar.
    """
    from tinyq.export.gguf_export import _special_token_ids

    vocab = {"hola": 0, "mundo": 1}
    added = {151645: "<|im_end|>"}
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "pad_token": "hola"}), encoding="utf-8"
    )

    ids = _special_token_ids(tmp_path, vocab, {v: k for k, v in added.items()})
    assert ids["eos"] == 151645
    assert ids["pad"] == 0


def test_pre_tokenizador_por_arquitectura(tmp_path):
    """Con el pre-tokenizador equivocado el texto se parte distinto a como el
    modelo aprendio, y la perplejidad sube aunque los pesos sean correctos."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from tinyq.export.gguf_export import _PRE_POR_ARCH

    assert _PRE_POR_ARCH["qwen2"] == "qwen2"

    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
    )
    model = LlamaForCausalLM(cfg)
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    _fake_tokenizer_json(tmp_path)
    out = export_gguf(model, tmp_path / "pre.gguf", tmp_path)

    reader = gguf.GGUFReader(str(out))
    assert reader.fields["tokenizer.ggml.pre"].contents() == "llama-bpe"
