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
    q = q.reshape(out_features, n_blocks, QK).to(torch.uint8).cpu().numpy()

    scales = scales.float().cpu()
    zeros = zeros.float().cpu()
    d = scales.numpy().astype(np.float16)
    m = (-zeros * scales).numpy().astype(np.float16)

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
    g = w.float().cpu().reshape(out_features, n_blocks, QK)
    d = (g.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(g / d), -127, 127).to(torch.int8).cpu().numpy()
    d16 = d.squeeze(-1).cpu().numpy().astype(np.float16)

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


def _rope_theta(cfg: Any) -> float:
    """Base de RoPE, la busque donde la busque cada version de transformers.

    A partir de transformers 5 el valor dejo de estar en `cfg.rope_theta` y
    vive dentro de `cfg.rope_parameters`. Un `getattr` con default se lo tragaba
    en silencio y escribia 10000 donde Qwen2.5 usa 1000000: el modelo seguia
    respondiendo frases cortas, pero la perplejidad se duplicaba en ventanas
    largas. Por eso aqui no hay default silencioso: si no aparece, se avisa.
    """
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        params = getattr(cfg, "rope_parameters", None) or {}
        if not isinstance(params, dict):
            params = getattr(params, "__dict__", {}) or {}
        theta = params.get("rope_theta") or params.get("theta")
    if theta is None:
        raise ValueError(
            "no se encontro rope_theta en la configuracion del modelo. "
            "Escribir un valor por defecto corrompe la atencion en silencio: "
            "revisa config.json (en transformers 5 esta en 'rope_parameters')."
        )
    return float(theta)
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
    writer.add_rope_freq_base(_rope_theta(cfg))
    writer.add_file_type(1)  # MOSTLY_F16 como base; los tensores llevan su tipo
    writer.add_quantization_version(2)  # GGML_QNT_VERSION: llama.cpp lo exige


def _write_vocab(writer, model_dir: Path, n_vocab: int = 0, arch: str = "") -> None:
    """Escribe el vocabulario BPE leyendo tokenizer.json de Hugging Face."""
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"falta {tok_path}: exporta primero el tokenizer con tinyq quantize"
        )
    data = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = data["model"]["vocab"]
    merges = data["model"].get("merges", [])

    added = {t["id"]: t for t in data.get("added_tokens", [])}
    # el modelo puede declarar mas tokens de los que trae el tokenizer: los
    # huecos se rellenan como UNUSED o llama.cpp rechaza el archivo
    size = max([len(vocab), n_vocab] + [i + 1 for i in added])
    tokens = [f"[UNUSED_{i}]" for i in range(size)]
    types = [5] * size  # UNUSED

    for tok, idx in vocab.items():
        tokens[idx] = tok
        types[idx] = 1  # NORMAL

    for idx, spec in added.items():
        tokens[idx] = spec["content"]
        types[idx] = 3 if spec.get("special") else 4  # CONTROL / USER_DEFINED

    writer.add_tokenizer_model("gpt2")
    # El pre-tokenizador decide como se parte el texto ANTES de buscar tokens.
    # Con "default" la misma frase produce tokens distintos a los que el modelo
    # vio al entrenarse: sigue respondiendo, pero la perplejidad se dispara.
    writer.add_tokenizer_pre(_PRE_POR_ARCH.get(arch, "default"))
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    if merges:
        writer.add_token_merges(
            [" ".join(m) if isinstance(m, list) else m for m in merges]
        )

    # tokens especiales: sin ellos llama.cpp no sabe cuando parar de generar
    ids = _special_token_ids(model_dir, vocab, {s["content"]: i for i, s in added.items()})
    if ids.get("bos") is not None:
        writer.add_bos_token_id(ids["bos"])
    if ids.get("eos") is not None:
        writer.add_eos_token_id(ids["eos"])
    if ids.get("pad") is not None:
        writer.add_pad_token_id(ids["pad"])
    writer.add_add_bos_token(ids.get("add_bos", False))


_PRE_POR_ARCH = {
    "qwen2": "qwen2",
    "llama": "llama-bpe",
    "gemma": "default",
}


def _special_token_ids(model_dir: Path, vocab: dict[str, int], extra: dict[str, int] | None = None) -> dict:
    """Lee bos/eos/pad de la configuracion del tokenizer de Hugging Face."""
    out: dict = {}
    extra = extra or {}
    cfg_path = model_dir / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        out["add_bos"] = bool(cfg.get("add_bos_token", False))
        for key, name in (("bos", "bos_token"), ("eos", "eos_token"), ("pad", "pad_token")):
            tok = cfg.get(name)
            if isinstance(tok, dict):
                tok = tok.get("content")
            # Los especiales de Qwen (<|im_end|>) no estan en model.vocab sino
            # en added_tokens: buscarlos solo en vocab los perdia en silencio.
            if isinstance(tok, str):
                idx = vocab.get(tok, extra.get(tok))
                if idx is not None:
                    out[key] = idx

    gen_path = model_dir / "generation_config.json"
    if gen_path.exists():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        for key, name in (("bos", "bos_token_id"), ("eos", "eos_token_id")):
            val = gen.get(name)
            if isinstance(val, list):
                val = val[0]
            if isinstance(val, int):
                out[key] = val
    return out


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
    _write_vocab(writer, model_dir, n_vocab=getattr(cfg, "vocab_size", 0), arch=arch)

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
        # el sesgo va aparte y sin cuantizar: Qwen2 lo usa en q, k y v, y sin
        # el la salida del modelo es basura
        if module.bias is not None:
            writer.add_tensor(
                f"{gg}.bias", module.bias.float().cpu().numpy().astype(np.float32)
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
            writer.add_tensor(f"{gg}.{kind}", t.cpu().numpy().astype(np.float32))
        written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out_file
