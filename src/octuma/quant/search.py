"""Busqueda de la escala que minimiza el error, en lugar de usar min/max.

Tomar el minimo y el maximo del grupo parece lo natural, pero un solo peso
atipico estira la escala y desperdicia niveles para todos los demas. Recortar
un poco ese rango casi siempre reduce el error total: se pierde precision en el
atipico y se gana en los otros 63 pesos del grupo.

Aqui se prueba una rejilla de factores de recorte y se queda el mejor por grupo,
midiendo el error cuadratico real de reconstruccion.
"""

from __future__ import annotations

import torch


def _params_from_range(
    vmin: torch.Tensor, vmax: torch.Tensor, qmax: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if symmetric:
        absmax = torch.maximum(vmax.abs(), vmin.abs())
        scale = (absmax / (qmax // 2)).clamp(min=1e-8)
        zero = torch.full_like(scale, float(qmax // 2))
    else:
        scale = ((vmax - vmin) / qmax).clamp(min=1e-8)
        zero = torch.round(-vmin / scale).clamp(0, qmax)
    return scale, zero


def _reconstruct(
    block: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, qmax: int
) -> torch.Tensor:
    q = torch.clamp(torch.round(block / scale) + zero, 0, qmax)
    return (q - zero) * scale


def search_group_params(
    block: torch.Tensor,
    bits: int,
    symmetric: bool = False,
    n_grid: int = 20,
    min_shrink: float = 0.45,
    norm: float = 2.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Escala y punto cero por fila de `block` [filas, group_size].

    `norm` > 2 penaliza mas los errores grandes que el simple error cuadratico;
    2.4 es el valor que usan varios trabajos de cuantizacion y da mejores
    resultados que 2.0 en la practica.
    """
    qmax = (1 << bits) - 1
    vmin = block.amin(dim=1, keepdim=True)
    vmax = block.amax(dim=1, keepdim=True)

    best_scale, best_zero = _params_from_range(vmin, vmax, qmax, symmetric)
    best_err = (_reconstruct(block, best_scale, best_zero, qmax) - block).abs().pow(norm).sum(
        dim=1, keepdim=True
    )

    for i in range(1, n_grid + 1):
        shrink = 1.0 - (1.0 - min_shrink) * i / n_grid
        scale, zero = _params_from_range(vmin * shrink, vmax * shrink, qmax, symmetric)
        err = (_reconstruct(block, scale, zero, qmax) - block).abs().pow(norm).sum(
            dim=1, keepdim=True
        )
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
        best_zero = torch.where(better, zero, best_zero)

    return best_scale, best_zero


def quantize_groupwise_searched(
    w: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
    symmetric: bool = False,
    n_grid: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Igual que `quantize_groupwise` pero buscando la escala por grupo."""
    out_features, in_features = w.shape
    if group_size <= 0:
        group_size = in_features
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} no es multiplo de group_size={group_size}"
        )

    qmax = (1 << bits) - 1
    n_groups = in_features // group_size
    g = w.float().reshape(out_features, n_groups, group_size)

    flat = g.reshape(out_features * n_groups, group_size)
    scale, zero = search_group_params(flat, bits, symmetric, n_grid=n_grid)
    scale = scale.reshape(out_features, n_groups)
    zero = zero.reshape(out_features, n_groups)

    q = torch.clamp(
        torch.round(g / scale.unsqueeze(-1)) + zero.unsqueeze(-1), 0, qmax
    ).reshape(w.shape)
    return q.to(torch.uint8), scale.half(), zero.to(torch.uint8)
