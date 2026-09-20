"""AWQ: escalado de canales consciente de las activaciones.

No todos los canales de entrada pesan igual. Si un canal recibe activaciones
grandes, su columna de pesos merece mas resolucion. AWQ multiplica esa columna
por un factor s > 1 antes de cuantizar (asi el redondeo le hace menos dano) y
divide por s en la operacion anterior, de modo que la funcion total no cambia:

    y = W·x = (W·diag(s)) · (diag(1/s)·x)

Como la division se pliega en la capa previa (una norma RMS o un lineal), en
inferencia no cuesta nada.

El exponente del escalado se busca: s = mean|x|^alpha, con alpha en [0, 1],
probando cual minimiza el error de salida real de la capa.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .core import quantize_tensor


@dataclass
class ScalingGroup:
    """Capas que comparten la misma entrada y pueden escalarse juntas."""

    name: str
    prev: nn.Module  # modulo donde se pliega 1/s (RMSNorm o Linear)
    layers: list[nn.Linear]
    prev_is_norm: bool


def llama_like_groups(block: nn.Module, n_heads: int, n_kv_heads: int) -> list[ScalingGroup]:
    """Grupos de escalado de un bloque tipo Llama / Qwen2 / Mistral."""
    groups: list[ScalingGroup] = []
    attn = getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None)
    if attn is None or mlp is None:
        return groups

    if hasattr(block, "input_layernorm"):
        groups.append(
            ScalingGroup(
                "attn_qkv",
                block.input_layernorm,
                [attn.q_proj, attn.k_proj, attn.v_proj],
                True,
            )
        )
    # o_proj solo se puede plegar en v_proj si no hay GQA: con menos cabezas de
    # value las dimensiones no corresponden una a una
    if n_heads == n_kv_heads:
        groups.append(ScalingGroup("attn_out", attn.v_proj, [attn.o_proj], False))

    if hasattr(block, "post_attention_layernorm"):
        groups.append(
            ScalingGroup(
                "mlp_in",
                block.post_attention_layernorm,
                [mlp.gate_proj, mlp.up_proj],
                True,
            )
        )
    groups.append(ScalingGroup("mlp_out", mlp.up_proj, [mlp.down_proj], False))
    return groups


@torch.no_grad()
def _group_output_error(
    layers: list[nn.Linear],
    x: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
) -> float:
    """Error de salida del grupo al cuantizar con el escalado `scales`."""
    total = 0.0
    for lin in layers:
        w = lin.weight.data.float()
        ref = x @ w.T
        w_scaled = w * scales.unsqueeze(0)
        qt = quantize_tensor(w_scaled, bits=bits, group_size=group_size)
        w_hat = qt.dequantize() / scales.unsqueeze(0)
        total += (x @ w_hat.T - ref).pow(2).sum().item()
    return total


@torch.no_grad()
def search_scales(
    layers: list[nn.Linear],
    x: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    n_grid: int = 12,
    max_samples: int = 2048,
) -> tuple[torch.Tensor, float, float]:
    """Busca alpha y devuelve (scales, alpha, error relativo conseguido)."""
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0])[:max_samples]
        x = x[idx]
    # las muestras pueden venir de CPU y los pesos estar en GPU
    x = x.float().to(layers[0].weight.device)

    act = x.abs().mean(dim=0).clamp(min=1e-5)
    act = act / act.mean()

    base = _group_output_error(
        layers, x, torch.ones_like(act), bits, group_size
    )
    best = (torch.ones_like(act), 0.0, base)

    for i in range(1, n_grid + 1):
        alpha = i / n_grid
        s = act.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()  # centrado: ni infla ni encoge el rango
        err = _group_output_error(layers, x, s, bits, group_size)
        if err < best[2]:
            best = (s, alpha, err)

    scales, alpha, err = best
    return scales, alpha, err / max(base, 1e-12)


@torch.no_grad()
def apply_scales(group: ScalingGroup, scales: torch.Tensor) -> None:
    """Multiplica los pesos del grupo por s y pliega 1/s en la capa previa."""
    scales = scales.to(group.layers[0].weight.device)
    for lin in group.layers:
        lin.weight.data = (lin.weight.data.float() * scales.unsqueeze(0)).to(
            lin.weight.dtype
        )

    prev = group.prev
    scales = scales.to(prev.weight.device)
    if group.prev_is_norm:
        prev.weight.data = (prev.weight.data.float() / scales).to(prev.weight.dtype)
    else:
        # lineal previo: cada fila de salida alimenta un canal de entrada
        prev.weight.data = (
            prev.weight.data.float() / scales.unsqueeze(1)
        ).to(prev.weight.dtype)
        if getattr(prev, "bias", None) is not None:
            prev.bias.data = (prev.bias.data.float() / scales).to(prev.bias.dtype)


@torch.no_grad()
def awq_scale_block(
    block: nn.Module,
    inputs_by_layer: dict[nn.Linear, torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    bits: int = 4,
    group_size: int = 64,
    n_grid: int = 12,
) -> tuple[list[dict], dict[nn.Linear, torch.Tensor]]:
    """Aplica AWQ a un bloque.

    Devuelve el reporte por grupo y el escalado aplicado a cada capa, que hace
    falta para corregir la Hessiana antes de GPTQ.
    """
    report: list[dict] = []
    applied: dict[nn.Linear, torch.Tensor] = {}
    for group in llama_like_groups(block, n_heads, n_kv_heads):
        x = inputs_by_layer.get(group.layers[0])
        if x is None or x.numel() == 0:
            continue
        scales, alpha, rel = search_scales(
            group.layers, x, bits=bits, group_size=group_size, n_grid=n_grid
        )
        if rel >= 1.0:  # el escalado no ayudo: se deja como estaba
            report.append({"group": group.name, "alpha": 0.0, "rel_err": 1.0})
            continue
        apply_scales(group, scales)
        for lin in group.layers:
            applied[lin] = scales
        report.append({"group": group.name, "alpha": alpha, "rel_err": rel})
    return report, applied
