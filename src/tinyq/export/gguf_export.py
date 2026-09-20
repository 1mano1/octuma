"""Exportacion a GGUF para llama.cpp (y de ahi a Android).

Mapeo clave: un grupo asimetrico de 32 pesos de TinyQ es exactamente un bloque
Q4_1 de llama.cpp.

    TinyQ:  w = (q - z) * s
    Q4_1:   w = d * q + m        =>   d = s,  m = -z * s

Por eso `--group 32` es obligatorio para exportar a GGUF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..quant.core import unpack_bits
from ..quant.qlinear import QuantLinear

QK = 32  # pesos por bloque en Q4_1 / Q8_0


def _q4_1_blocks(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> np.ndarray:
    """Empaqueta [out, in] en bloques Q4_1: d(f16), m(f16), 16 bytes."""
    out_features, in_features = q.shape
    n_blocks = in_features // QK
    q = q.reshape(out_features, n_blocks, QK).to(torch.uint8).numpy()

    d = scales.float().numpy().astype(np.float16)
    m = (-zeros.float() * scales.float()).numpy().astype(np.float16)

    # llama.cpp guarda el valor j en el nibble bajo y el j+16 en el alto
    low = q[:, :, :16]
    high = q[:, :, 16:]
    qs = (low | (high << 4)).astype(np.uint8)

    blocks = np.empty((out_features, n_blocks, 20), dtype=np.uint8)
    blocks[:, :, 0:2] = d.view(np.uint8).reshape(out_features, n_blocks, 2)
    blocks[:, :, 2:4] = m.view(np.uint8).reshape(out_features, n_blocks, 2)
    blocks[:, :, 4:20] = qs
    return blocks.reshape(out_features, n_blocks * 20)


def _q8_0_blocks(w: torch.Tensor) -> np.ndarray:
    """Cuantiza a Q8_0 (simetrico, bloque de 32): d(f16) + 32 int8."""
    out_features, in_features = w.shape
    n_blocks = in_features // QK
    g = w.float().reshape(out_features, n_blocks, QK)
    d = (g.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(g / d), -127, 127).to(torch.int8).numpy()
    d16 = d.squeeze(-1).numpy().astype(np.float16)

    blocks = np.empty((out_features, n_blocks, 34), dtype=np.uint8)
    blocks[:, :, 0:2] = d16.view(np.uint8).reshape(out_features, n_blocks, 2)
    blocks[:, :, 2:34] = q.view(np.uint8)
    return blocks.reshape(out_features, n_blocks * 34)


def _arch_for(model_type: str) -> str:
    known = {"llama": "llama", "qwen2": "qwen2", "mistral": "llama", "gemma": "gemma"}
    if model_type not in known:
        raise ValueError(
            f"arquitectura '{model_type}' no soportada todavia en el exportador GGUF"
        )
    return known[model_type]


def _write_metadata(writer, cfg: Any, arch: str, name: str) -> None:
    writer.add_name(name)
    writer.add_context_length(getattr(cfg, "max_position_embeddings", 2048))
    writer.add_embedding_length(cfg.hidden_size)
    writer.add_block_count(cfg.num_hidden_layers)
    writer.add_feed_forward_length(cfg.intermediate_size)
    writer.add_head_count(cfg.num_attention_heads)
    writer.add_head_count_kv(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    writer.add_layer_norm_rms_eps(getattr(cfg, "rms_norm_eps", 1e-6))
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(getattr(cfg, "rope_theta", 10000.0))
    writer.add_file_type(1)  # MOSTLY_F16 como base; los tensores llevan su tipo


def _write_vocab(writer, model_dir: Path) -> None:
    """Escribe el vocabulario BPE leyendo tokenizer.json de Hugging Face."""
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"falta {tok_path}: exporta primero el tokenizer con tinyq quantize"
        )
    data = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = data["model"]["vocab"]
    merges = data["model"].get("merges", [])

    tokens = [""] * len(vocab)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    types = [1] * len(tokens)  # NORMAL

    added = {t["id"]: t for t in data.get("added_tokens", [])}
    for idx, spec in added.items():
        if idx < len(tokens):
            tokens[idx] = spec["content"]
            types[idx] = 3 if spec.get("special") else 1  # CONTROL / NORMAL

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    if merges:
        writer.add_token_merges(
            [" ".join(m) if isinstance(m, list) else m for m in merges]
        )


def export_gguf(
    model: nn.Module,
    out_file: str | Path,
    model_dir: str | Path,
    name: str = "tinyq-model",
) -> Path:
    """Escribe un GGUF con los pesos cuantizados de TinyQ.

    `model_dir` es la carpeta .tq (necesita tokenizer.json y config).
    """
    import gguf

    out_file = Path(out_file)
    model_dir = Path(model_dir)
    cfg = model.config
    arch = _arch_for(cfg.model_type)

    qlayers = {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}
    bad = [n for n, m in qlayers.items() if m.group_size != QK or m.bits != 4]
    if bad:
        raise ValueError(
            "GGUF Q4_1 requiere --bits 4 --group 32; capas incompatibles: "
            + ", ".join(bad[:3])
        )

    writer = gguf.GGUFWriter(str(out_file), arch)
    _write_metadata(writer, cfg, arch, name)
    _write_vocab(writer, model_dir)

    arch_enum = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}[arch]
    name_map = gguf.TensorNameMap(arch_enum, cfg.num_hidden_layers)

    state = model.state_dict()
    written = 0

    for hf_name, module in qlayers.items():
        gg = name_map.get_name(hf_name)
        if gg is None:
            raise ValueError(f"no se encontro el nombre GGUF para {hf_name}")
        q = unpack_bits(module.qweight, module.bits, module.in_features)
        data = _q4_1_blocks(q, module.scales, module.zeros)
        writer.add_tensor(
            gg + ".weight", data, raw_dtype=gguf.GGMLQuantizationType.Q4_1
        )
        written += 1

    for hf_name, tensor in state.items():
        base, _, kind = hf_name.rpartition(".")
        if base in qlayers or kind not in ("weight", "bias"):
            continue
        gg = name_map.get_name(base)
        if gg is None:
            continue
        t = tensor.float().cpu()
        if kind == "weight" and t.ndim == 2 and t.shape[1] % QK == 0 and t.numel() > 1_000_000:
            writer.add_tensor(
                f"{gg}.weight", _q8_0_blocks(t), raw_dtype=gguf.GGMLQuantizationType.Q8_0
            )
        else:
            writer.add_tensor(f"{gg}.{kind}", t.numpy().astype(np.float32))
        written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out_file
