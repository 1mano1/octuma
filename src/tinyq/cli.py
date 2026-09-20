"""Interfaz de linea de comandos de TinyQ."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    add_completion=False,
    help="TinyQ: cuantiza modelos de lenguaje a INT4/INT8 y los deja listos para equipos modestos y Android.",
)
console = Console()


def _load_model(model_id: str, device: str, dtype: str = "float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tok


@app.command()
def version() -> None:
    """Muestra la version instalada."""
    console.print(f"TinyQ {__version__}")


@app.command()
def quantize(
    model: str = typer.Argument(..., help="Id de Hugging Face o carpeta local"),
    out: Path = typer.Option(..., "--out", "-o", help="Carpeta de salida"),
    bits: int = typer.Option(4, help="Bits por peso: 2, 3, 4 u 8"),
    group_size: int = typer.Option(64, "--group", help="Pesos por grupo de escala"),
    method: str = typer.Option("gptq", help="gptq (con calibracion) o rtn (directo)"),
    calib: str = typer.Option("wikitext2", help="Dataset de calibracion o ruta a textos"),
    samples: int = typer.Option(64, help="Ventanas de calibracion"),
    seq_len: int = typer.Option(512, "--seqlen", help="Tokens por ventana"),
    symmetric: bool = typer.Option(False, help="Cuantizacion simetrica"),
    device: str = typer.Option("cpu", help="cpu o cuda"),
    dtype: str = typer.Option("float32", help="Precision de carga"),
) -> None:
    """Calibra, cuantiza y guarda el modelo en formato .tq."""
    from .calibrate import load_calibration
    from .export.tq import disk_size, save_quantized
    from .quantizer import QuantConfig, quantize_model

    console.print(f"[bold]Cargando[/bold] {model} ({dtype}, {device})")
    net, tok = _load_model(model, device, dtype)

    console.print(f"[bold]Calibrando[/bold] con {calib}: {samples} x {seq_len} tokens")
    cal = load_calibration(calib, tok, n_samples=samples, seq_len=seq_len)

    cfg = QuantConfig(bits=bits, group_size=group_size, method=method, symmetric=symmetric)
    console.print(f"[bold]Cuantizando[/bold] a INT{bits}, grupos de {group_size} ({method})")
    report = quantize_model(net, cal, cfg, device=device, progress=console.print)

    save_quantized(net, out, cfg=cfg, extra={"source_model": model, "calibration": cal.source})
    tok.save_pretrained(out)
    if hasattr(net, "config"):
        net.config.save_pretrained(out)

    s = report.summary()
    table = Table(title="Resultado", show_header=False)
    table.add_row("Capas cuantizadas", str(s["n_layers"]))
    table.add_row("Pesos FP16", f"{s['fp_gb']:.2f} GB")
    table.add_row("Pesos INT" + str(bits), f"{s['q_gb']:.2f} GB")
    table.add_row("Compresion", f"{s['compression']:.2f}x")
    table.add_row("Error medio", f"{s['mean_rel_fro']:.4f}")
    table.add_row("Peor capa", f"{s['worst_layer'][0]} ({s['worst_layer'][1]:.4f})")
    table.add_row("Tiempo", f"{s['seconds'] / 60:.1f} min")
    table.add_row("En disco", f"{disk_size(out) / 1e9:.2f} GB")
    console.print(table)
    console.print(f"[green]Listo[/green] -> {out}")


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="Modelo HF o carpeta .tq"),
    dataset: str = typer.Option("wikitext2", help="Dataset de evaluacion"),
    windows: int = typer.Option(20, help="Ventanas a evaluar"),
    seq_len: int = typer.Option(512, "--seqlen"),
    device: str = typer.Option("cpu"),
    speed: bool = typer.Option(False, help="Mide tokens por segundo"),
    out: Path = typer.Option(None, "--out", "-o", help="Guarda el reporte en JSON"),
) -> None:
    """Mide perplejidad, memoria y velocidad."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .evaluate import generation_speed, model_size_bytes, perplexity, wikitext2_ids
    from .export.tq import load_quantized

    tq_dir = Path(model)
    is_tq = (tq_dir / "tinyq.json").exists()

    if is_tq:
        console.print(f"[bold]Cargando modelo cuantizado[/bold] {model}")
        cfg = AutoConfig.from_pretrained(model)
        net = AutoModelForCausalLM.from_config(cfg)
        net = load_quantized(net, tq_dir, device=device)
        net.to(device).eval()
        tok = AutoTokenizer.from_pretrained(model)
    else:
        net, tok = _load_model(model, device)

    ids = wikitext2_ids(tok) if dataset.startswith("wikitext") else None
    if ids is None:
        raise typer.BadParameter(f"dataset no soportado: {dataset}")

    res = perplexity(
        net, ids, seq_len=seq_len, device=device, dataset=dataset,
        max_windows=windows, progress=None,
    )
    sizes = model_size_bytes(net)

    table = Table(title=f"Evaluacion · {model}", show_header=False)
    table.add_row("Perplejidad", f"{res.perplexity:.3f}")
    table.add_row("Ventanas", f"{res.n_windows} x {res.seq_len} tokens")
    table.add_row("Pesos cuantizados", f"{sizes['quantized'] / 1e9:.2f} GB")
    table.add_row("Pesos densos", f"{sizes['dense'] / 1e9:.2f} GB")
    table.add_row("Embeddings", f"{sizes['embeddings'] / 1e9:.2f} GB")
    table.add_row("Total en memoria", f"{sizes['total'] / 1e9:.2f} GB")
    table.add_row("Tiempo", f"{res.seconds:.1f} s")

    payload = {"model": model, "perplexity": res.perplexity, "sizes": sizes,
               "windows": res.n_windows, "seq_len": res.seq_len}

    if speed:
        tps = generation_speed(net, ids[:, :64], n_tokens=16, device=device)
        table.add_row("Generacion", f"{tps:.2f} tok/s")
        payload["tokens_per_second"] = tps

    console.print(table)
    if out:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Reporte -> {out}")


@app.command()
def export(
    model_dir: Path = typer.Argument(..., help="Carpeta .tq generada por quantize"),
    out: Path = typer.Option(..., "--out", "-o", help="Archivo .gguf de salida"),
    name: str = typer.Option("tinyq-model", help="Nombre dentro del GGUF"),
) -> None:
    """Convierte un modelo .tq a GGUF para llama.cpp y Android."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .export.gguf_export import export_gguf
    from .export.tq import load_quantized

    console.print(f"[bold]Cargando[/bold] {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir)
    net = AutoModelForCausalLM.from_config(cfg)
    net = load_quantized(net, model_dir)

    console.print("[bold]Escribiendo GGUF[/bold] (Q4_1 para los pesos cuantizados)")
    path = export_gguf(net, out, model_dir, name=name)
    console.print(f"[green]Listo[/green] -> {path} ({path.stat().st_size / 1e9:.2f} GB)")
    console.print("Pruebalo con: llama-cli -m " + str(path) + " -p \"Hola\"")


@app.command()
def info(model_dir: Path = typer.Argument(..., help="Carpeta .tq")) -> None:
    """Muestra el contenido de un modelo cuantizado."""
    meta = json.loads((model_dir / "tinyq.json").read_text(encoding="utf-8"))
    layers = meta["layers"]
    bits = {}
    for spec in layers.values():
        bits[spec["bits"]] = bits.get(spec["bits"], 0) + 1

    table = Table(title=f"TinyQ · {model_dir.name}", show_header=False)
    table.add_row("Modelo origen", str(meta.get("source_model", "?")))
    table.add_row("Arquitectura", str(meta.get("model_type", "?")))
    table.add_row("Calibracion", str(meta.get("calibration", "?")))
    table.add_row("Capas cuantizadas", str(len(layers)))
    table.add_row("Bits", ", ".join(f"INT{b}: {n} capas" for b, n in sorted(bits.items())))
    cfg = meta.get("config", {})
    if cfg:
        table.add_row("Grupo", str(cfg.get("group_size")))
        table.add_row("Metodo", str(cfg.get("method")))
    console.print(table)


if __name__ == "__main__":
    app()
