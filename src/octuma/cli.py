"""Interfaz de linea de comandos de Octuma."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .logo import anotar_version, dibujar, saludar_si_es_nueva, terminal_interactiva

app = typer.Typer(
    add_completion=False,
    help="Octuma: quantizes language models to INT4/INT8 so they run on modest hardware and Android.",
)
console = Console()


def _imprimir_version() -> None:
    # en una tuberia sale solo la linea, para que la puedan leer los scripts
    if terminal_interactiva():
        dibujar(console, __version__)
        anotar_version()
    else:
        console.print(f"Octuma {__version__}")


def _mostrar_version(pedida: bool) -> None:
    if pedida:
        _imprimir_version()
        raise typer.Exit()


@app.callback()
def _principal(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_mostrar_version,
        is_eager=True,
        help="Show the installed version and exit",
    ),
) -> None:
    """Punto de entrada: solo existe para colgar de el la bandera --version.

    El subcomando `octuma version` hace lo mismo y se queda por compatibilidad,
    pero lo que la gente teclea sin pensar es `--version`.

    Tambien dibuja el pulpo la primera vez que se usa una version nueva.
    """
    saludar_si_es_nueva(console)


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




def _resolver_device_dtype(device: str, dtype: str, avisar_cpu: bool = True) -> tuple[str, str]:
    """Elige GPU y precision solos.

    Los valores por defecto de antes (cpu + float32) hacian que el comando mas
    obvio tardara horas y diera la impresion de que Octuma es lento.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        # en CPU float16 va lentisimo o ni siquiera esta soportado
        dtype = "float16" if device.startswith("cuda") else "float32"
    if device.startswith("cuda") and not torch.cuda.is_available():
        console.print("[yellow]No GPU available; using CPU[/yellow]")
        device, dtype = "cpu", "float32"
    if device == "cpu" and avisar_cpu:
        console.print(
            "[yellow]No GPU: this will take a while on CPU.[/yellow] "
            "For a large model use a GPU, or try --bits 8."
        )
    return device, dtype


def _leer_meta(model_dir: Path) -> dict:
    """Lee el octuma.json de una carpeta .tq, o explica por que no puede.

    Sin esto, pasar una carpeta que no existe daba un FileNotFoundError crudo
    en `info` y `compare`, y algo peor en `try` y `export`: transformers toma
    el nombre por un repo de Hugging Face, va a buscarlo y devuelve un 401 de
    veinte lineas. El usuario acababa leyendo sobre tokens de autenticacion
    cuando el problema era que `quantize` no habia llegado a escribir nada.
    """
    if not model_dir.exists():
        raise typer.BadParameter(
            f"folder '{model_dir}' does not exist. If you just quantized, "
            "check that 'octuma quantize' finished: if it stopped halfway "
            "it writes nothing."
        )
    if not (model_dir / "octuma.json").exists():
        raise typer.BadParameter(
            f"'{model_dir}' exists but has no octuma.json, so it is not "
            "a folder quantized by Octuma."
        )
    return json.loads((model_dir / "octuma.json").read_text(encoding="utf-8"))


def _memoria_libre() -> tuple[float | None, float | None]:
    """RAM y VRAM libres, en GB. Devuelve None en la que no se pueda medir.

    La VRAM sale de `mem_get_info()[0]`, que es lo que queda de verdad. Antes
    se leia `total_memory` y se imprimia con la palabra "libres": el aviso
    siempre sonaba holgado porque no descontaba nada de lo ya ocupado.
    """
    import torch

    try:
        import psutil

        ram = psutil.virtual_memory().available / 1e9
    except Exception:
        ram = None
    vram = None
    if torch.cuda.is_available():
        try:
            vram = torch.cuda.mem_get_info()[0] / 1e9
        except Exception:
            vram = None
    return ram, vram


def _avisar_memoria(model: str, device: str) -> None:
    """Compara lo que pide el modelo con lo que hay, ANTES de descargarlo.

    Reventar por falta de memoria a los diez minutos, despues de bajar 15 GB,
    es la peor primera impresion posible.

    Mira las dos memorias aunque se cuantice en GPU. Antes, con CUDA presente,
    solo miraba la VRAM: imprimia "la GPU tiene 8.6 GB libres" justo antes de
    morir por falta de RAM, que es donde se acumulan las muestras de AWQ. Un
    aviso que no mide lo que falla es peor que no avisar, porque da confianza.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, files_metadata=True)
        pesos = sum(
            s.size or 0 for s in (info.siblings or [])
            if s.rfilename.endswith((".safetensors", ".bin"))
        )
    except Exception:
        return  # es una carpeta local, o no hay red: no estorbamos
    if not pesos:
        return

    # el modelo cargado mas las entradas de calibracion, que viven donde corre
    necesario = pesos * 1.4 / 1e9
    ram, vram = _memoria_libre()
    en_gpu = device.startswith("cuda") and vram is not None

    partes = []
    if ram is not None:
        partes.append(f"{ram:.1f} GB of RAM")
    if vram is not None:
        partes.append(f"{vram:.1f} GB of VRAM")
    if not partes:
        return
    console.print(
        f"This model needs ~{necesario:.1f} GB and you have free: " + " and ".join(partes)
    )

    corto = vram if en_gpu else ram
    if corto is not None and necesario > corto:
        console.print(
            "[yellow]It may not fit.[/yellow] Options: --bits 8, a smaller "
            "model, or free up memory."
        )
    # cuantizar con AWQ guarda muestras de activaciones en RAM aunque el modelo
    # este en la GPU. No depende del tamaño del modelo, asi que se avisa aparte.
    if ram is not None and ram < 4.0:
        console.print(
            f"[yellow]Only {ram:.1f} GB of RAM left.[/yellow] Quantizing with AWQ "
            "needs RAM even when the model runs on the GPU; if it runs short, "
            "use --no-awq or close something."
        )


@app.command()
def version() -> None:
    """Show the installed version."""
    _imprimir_version()


@app.command()
def quantize(
    model: str = typer.Argument(..., help="Hugging Face id or local folder"),
    out: Path = typer.Option(None, "--out", "-o", help="Output folder (derived from the model by default)"),
    bits: int = typer.Option(4, help="Bits per weight: 2, 3, 4 or 8"),
    group_size: int = typer.Option(32, "--group", help="Weights per scale group"),
    method: str = typer.Option("gptq", help="gptq (with calibration) or rtn (direct)"),
    calib: str = typer.Option("wikitext2", help="Calibration dataset or path to text files"),
    samples: int = typer.Option(128, help="Calibration windows"),
    seq_len: int = typer.Option(2048, "--seqlen", help="Tokens per window"),
    symmetric: bool = typer.Option(False, help="Symmetric quantization"),
    awq: bool = typer.Option(True, help="AWQ scaling before quantizing"),
    search_scale: bool = typer.Option(True, help="Search the best scale per group"),
    device: str = typer.Option("auto", help="auto, cpu or cuda"),
    dtype: str = typer.Option("auto", help="auto, float16, bfloat16 or float32"),
    plan: Path = typer.Option(
        None, "--plan", help="Mixed-precision JSON from 'octuma analyze'"
    ),
) -> None:
    """Calibrate, quantize and save the model in .tq format."""
    from .calibrate import load_calibration
    from .export.tq import disk_size, save_quantized
    from .quantizer import QuantConfig, quantize_model

    device, dtype = _resolver_device_dtype(device, dtype)
    if out is None:
        # sin --out: "Qwen/Qwen2.5-3B-Instruct" -> "qwen2.5-3b-instruct-int4"
        out = Path(f"{model.rstrip('/').split('/')[-1].lower()}-int{bits}")
        console.print(f"Output folder: [bold]{out}[/bold]")
    _avisar_memoria(model, device)

    console.print(f"[bold]Loading[/bold] {model} ({dtype}, {device})")
    net, tok = _load_model(model, device, dtype)

    console.print(f"[bold]Calibrating[/bold] with {calib}: {samples} x {seq_len} tokens")
    cal = load_calibration(calib, tok, n_samples=samples, seq_len=seq_len)

    overrides: dict[str, int] = {}
    if plan:
        raw = json.loads(plan.read_text(encoding="utf-8"))
        overrides = {k.split(".", 2)[-1]: int(v) for k, v in raw.items()}
        console.print(f"Mixed precision: {len(overrides)} patterns at more bits")

    cfg = QuantConfig(
        bits=bits,
        group_size=group_size,
        method=method,
        symmetric=symmetric,
        bits_overrides=overrides,
        awq=awq,
        search_scale=search_scale,
    )
    extras = ("AWQ + " if awq else "") + method
    console.print(f"[bold]Quantizing[/bold] to INT{bits}, groups of {group_size} ({extras})")
    report = quantize_model(net, cal, cfg, device=device, progress=console.print)

    save_quantized(net, out, cfg=cfg, extra={"source_model": model, "calibration": cal.source})
    tok.save_pretrained(out)
    if hasattr(net, "config"):
        net.config.save_pretrained(out)

    s = report.summary()
    table = Table(title="Result", show_header=False)
    table.add_row("Quantized layers", str(s["n_layers"]))
    table.add_row("FP16 weights", f"{s['fp_gb']:.2f} GB")
    table.add_row("INT" + str(bits) + " weights", f"{s['q_gb']:.2f} GB")
    table.add_row("Compression", f"{s['compression']:.2f}x")
    table.add_row("Mean error", f"{s['mean_rel_fro']:.4f}")
    table.add_row("Worst layer", f"{s['worst_layer'][0]} ({s['worst_layer'][1]:.4f})")
    table.add_row("Time", f"{s['seconds'] / 60:.1f} min")
    table.add_row("On disk", f"{disk_size(out) / 1e9:.2f} GB")
    console.print(table)
    console.print(f"[green]Done[/green] -> {out}")


@app.command()
def evaluate(
    model: str = typer.Argument(..., help="HF model or .tq folder"),
    dataset: str = typer.Option("wikitext2", help="Evaluation dataset"),
    windows: int = typer.Option(20, help="Windows to evaluate"),
    seq_len: int = typer.Option(512, "--seqlen"),
    device: str = typer.Option("cpu"),
    speed: bool = typer.Option(False, help="Measure tokens per second"),
    out: Path = typer.Option(None, "--out", "-o", help="Save the report as JSON"),
) -> None:
    """Measure perplexity, memory and speed."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .evaluate import generation_speed, model_size_bytes, perplexity, wikitext2_ids
    from .export.tq import load_quantized

    tq_dir = Path(model)
    is_tq = (tq_dir / "octuma.json").exists()

    if is_tq:
        console.print(f"[bold]Loading quantized model[/bold] {model}")
        cfg = AutoConfig.from_pretrained(model)
        net = AutoModelForCausalLM.from_config(cfg)
        net = load_quantized(net, tq_dir, device=device)
        net.to(device).eval()
        tok = AutoTokenizer.from_pretrained(model)
    else:
        net, tok = _load_model(model, device)

    ids = wikitext2_ids(tok) if dataset.startswith("wikitext") else None
    if ids is None:
        raise typer.BadParameter(f"unsupported dataset: {dataset}")

    res = perplexity(
        net, ids, seq_len=seq_len, device=device, dataset=dataset,
        max_windows=windows, progress=None,
    )
    sizes = model_size_bytes(net)

    table = Table(title=f"Evaluation · {model}", show_header=False)
    table.add_row("Perplexity", f"{res.perplexity:.3f}")
    table.add_row("Windows", f"{res.n_windows} x {res.seq_len} tokens")
    table.add_row("Quantized weights", f"{sizes['quantized'] / 1e9:.2f} GB")
    table.add_row("Dense weights", f"{sizes['dense'] / 1e9:.2f} GB")
    table.add_row("Embeddings", f"{sizes['embeddings'] / 1e9:.2f} GB")
    table.add_row("Total in memory", f"{sizes['total'] / 1e9:.2f} GB")
    table.add_row("Time", f"{res.seconds:.1f} s")

    payload = {"model": model, "perplexity": res.perplexity, "sizes": sizes,
               "windows": res.n_windows, "seq_len": res.seq_len}

    if speed:
        tps = generation_speed(net, ids[:, :64], n_tokens=16, device=device)
        table.add_row("Generation", f"{tps:.2f} tok/s")
        payload["tokens_per_second"] = tps

    console.print(table)
    if out:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Report -> {out}")


@app.command()
def analyze(
    model: str = typer.Argument(..., help="Hugging Face id or local folder"),
    calib: str = typer.Option("wikitext2", help="Calibration dataset"),
    samples: int = typer.Option(16, help="Calibration windows"),
    seq_len: int = typer.Option(256, "--seqlen"),
    group_size: int = typer.Option(64, "--group"),
    target_bits: float = typer.Option(4.5, help="Target average bits"),
    device: str = typer.Option("cpu"),
    out: Path = typer.Option(None, "--out", "-o", help="Save the plan as JSON"),
) -> None:
    """Find the layers that suffer most at 4 bits and propose a mixed-precision plan."""
    from .calibrate import load_calibration
    from .sensitivity import analyze_sensitivity

    console.print(f"[bold]Loading[/bold] {model}")
    net, tok = _load_model(model, device)
    cal = load_calibration(calib, tok, n_samples=samples, seq_len=seq_len)

    console.print("[bold]Analyzing sensitivity[/bold] (output error per layer)")
    report = analyze_sensitivity(
        net, cal, bits_options=(4, 8), group_size=group_size,
        device=device, progress=console.print,
    )

    table = Table(title="Most sensitive layers")
    table.add_column("Layer")
    table.add_column("INT4 error", justify="right")
    table.add_column("INT8 error", justify="right")
    for name, e4, e8 in report.table(top=12):
        table.add_row(name, f"{e4:.5f}", f"{e8:.5f}")
    console.print(table)

    plan = report.plan_mixed_precision(target_avg_bits=target_bits)
    console.print(f"Plan: {len(plan)} layers at INT8 for an average of {target_bits} bits")
    if out:
        out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        console.print(f"Plan -> {out}  (use it with: octuma quantize ... --plan {out})")


@app.command()
def export(
    model_dir: Path = typer.Argument(..., help=".tq folder made by quantize"),
    out: Path = typer.Option(..., "--out", "-o", help="Output .gguf file"),
    name: str = typer.Option("octuma-model", help="Name inside the GGUF"),
) -> None:
    """Convert a .tq model to GGUF for llama.cpp and Android."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .export.gguf_export import export_gguf
    from .export.tq import load_quantized

    _leer_meta(model_dir)  # antes de que transformers lo tome por un repo del Hub
    console.print(f"[bold]Loading[/bold] {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir)
    net = AutoModelForCausalLM.from_config(cfg)
    net = load_quantized(net, model_dir)

    console.print("[bold]Writing GGUF[/bold] (Q4_1 for the quantized weights)")
    path = export_gguf(net, out, model_dir, name=name)
    console.print(f"[green]Done[/green] -> {path} ({path.stat().st_size / 1e9:.2f} GB)")
    console.print("Try it with: llama-cli -m " + str(path) + " -p \"Hello\"")


@app.command()
def info(model_dir: Path = typer.Argument(..., help=".tq folder")) -> None:
    """Show what is inside a quantized model."""
    meta = _leer_meta(model_dir)
    layers = meta["layers"]
    bits = {}
    for spec in layers.values():
        bits[spec["bits"]] = bits.get(spec["bits"], 0) + 1

    table = Table(title=f"Octuma · {model_dir.name}", show_header=False)
    table.add_row("Source model", str(meta.get("source_model", "?")))
    table.add_row("Architecture", str(meta.get("model_type", "?")))
    table.add_row("Calibration", str(meta.get("calibration", "?")))
    table.add_row("Quantized layers", str(len(layers)))
    table.add_row("Bits", ", ".join(f"INT{b}: {n} layers" for b, n in sorted(bits.items())))
    cfg = meta.get("config", {})
    if cfg:
        table.add_row("Group", str(cfg.get("group_size")))
        table.add_row("Method", str(cfg.get("method")))
    console.print(table)


@app.command()
def compare(
    model_dir: Path = typer.Argument(..., help=".tq folder made by quantize"),
    original: str = typer.Option(None, help="Unquantized model (defaults to the one in octuma.json)"),
    windows: int = typer.Option(20, help="Windows to evaluate"),
    seq_len: int = typer.Option(2048, "--seqlen"),
    device: str = typer.Option("auto", help="auto, cpu or cuda"),
    speed: bool = typer.Option(True, help="Also measure tokens per second"),
) -> None:
    """Compare the quantized model with the original: did it come out well?"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .evaluate import generation_speed, model_size_bytes, perplexity, wikitext2_ids
    from .export.tq import load_quantized

    device, _ = _resolver_device_dtype(device, "auto", avisar_cpu=False)
    meta = _leer_meta(model_dir)
    original = original or meta.get("source_model")
    if not original:
        raise typer.BadParameter(
            "unknown source model for this .tq: pass --original"
        )

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    ids = wikitext2_ids(tok)
    filas = []

    for etiqueta, cargar in (
        ("Original", lambda: AutoModelForCausalLM.from_pretrained(
            original, dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True).to(device).eval()),
        ("Quantized", lambda: load_quantized(
            AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir)),
            model_dir, device=device).to(device).eval()),
    ):
        console.print(f"[bold]Measuring[/bold] {etiqueta.lower()}...")
        net = cargar()
        res = perplexity(net, ids, seq_len=seq_len, device=device,
                         dataset="wikitext2", max_windows=windows)
        mem = model_size_bytes(net)["total"] / 1e9
        tps = generation_speed(net, ids[:, :32], device=device) if speed else None
        filas.append((etiqueta, res.perplexity, mem, tps))
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = Table(title=f"{original} · {windows} windows of {seq_len}")
    table.add_column("")
    table.add_column("Perplexity", justify="right")
    table.add_column("Memory", justify="right")
    if speed:
        table.add_column("Speed", justify="right")
    for etiqueta, ppl, mem, tps in filas:
        fila = [etiqueta, f"{ppl:.3f}", f"{mem:.2f} GB"]
        if speed:
            fila.append(f"{tps:.1f} tok/s" if tps else "-")
        table.add_row(*fila)
    console.print(table)

    base, quant = filas[0], filas[1]
    perdida = (quant[1] / base[1] - 1) * 100
    console.print(
        f"[bold]{base[2] / quant[2]:.2f}x smaller[/bold] for "
        f"[bold]{perdida:+.1f}%[/bold] perplexity"
    )


PREGUNTAS_PRUEBA = [
    "Explain in two sentences what model quantization is.",
    "What is the capital of Australia?",
    "Write a Python function that reverses a string.",
    "What is 17 times 24? Show your work.",
    "Translate to Spanish: 'The cat sleeps by the window'.",
    "Which is heavier, a kilo of lead or a kilo of feathers?",
    "Write a haiku about rain.",
    "In what year did humans land on the Moon?",
    "Explain the difference between a list and a tuple in Python.",
    "Finish the saying: 'A bird in the hand...'",
]


def _responder(net, tok, pregunta: str, max_new: int, device: str) -> str:
    import torch

    texto = tok.apply_chat_template(
        [{"role": "user", "content": pregunta}], tokenize=False, add_generation_prompt=True
    )
    entrada = tok(texto, return_tensors="pt").to(device)
    torch.manual_seed(0)
    with torch.no_grad():
        salida = net.generate(**entrada, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    nuevos = salida[0][entrada.input_ids.shape[1]:]
    return tok.decode(nuevos, skip_special_tokens=True).strip()


@app.command("try")
def probar(
    model_dir: Path = typer.Argument(..., help=".tq folder made by quantize"),
    side_by_side: bool = typer.Option(
        False, "--side-by-side", help="Compare the answers with the original's"
    ),
    prompt: str = typer.Option(None, "-p", help="Ask one question and exit"),
    max_new: int = typer.Option(120, help="Tokens per answer"),
    device: str = typer.Option("auto", help="auto, cpu or cuda"),
    out: Path = typer.Option(None, "--out", "-o", help="Save the comparison as Markdown"),
) -> None:
    """Try the quantized model: does it still talk well?

    Perplexity can't answer that. Reading the answers can.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .export.tq import load_quantized

    device, _ = _resolver_device_dtype(device, "auto", avisar_cpu=False)
    _leer_meta(model_dir)  # antes de que transformers lo tome por un repo del Hub
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    console.print("[bold]Loading[/bold] the quantized model...")
    net = load_quantized(
        AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir)),
        model_dir, device=device,
    ).to(device).eval()

    if prompt:
        console.print(f"\n[bold cyan]{prompt}[/bold cyan]\n")
        console.print(_responder(net, tok, prompt, max_new, device))
        return

    if not side_by_side:
        # chat sencillo en la terminal
        console.print("Type your question ([dim]Ctrl+C to quit[/dim])\n")
        while True:
            try:
                pregunta = typer.prompt(">")
            except (KeyboardInterrupt, EOFError):
                console.print("\nBye")
                return
            console.print(f"\n{_responder(net, tok, pregunta, max_new, device)}\n")

    meta = json.loads((model_dir / "octuma.json").read_text(encoding="utf-8"))
    origen = meta.get("source_model")
    if not origen:
        raise typer.BadParameter("the .tq does not say which model it came from")

    respuestas = []
    for p in PREGUNTAS_PRUEBA:
        respuestas.append({"pregunta": p, "cuantizado": _responder(net, tok, p, max_new, device)})
        console.print(f"  [dim]quantized:[/dim] {p[:45]}...")
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print("[bold]Loading[/bold] the original to compare...")
    orig = AutoModelForCausalLM.from_pretrained(
        origen, dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    iguales = 0
    for fila in respuestas:
        fila["original"] = _responder(orig, tok, fila["pregunta"], max_new, device)
        if fila["original"].strip() == fila["cuantizado"].strip():
            iguales += 1
        console.print(f"  [dim]original:[/dim] {fila['pregunta'][:45]}...")

    for fila in respuestas:
        console.print(f"\n[bold cyan]{fila['pregunta']}[/bold cyan]")
        console.print(f"[dim]original  [/dim] {fila['original'][:300]}")
        console.print(f"[dim]quantized [/dim] {fila['cuantizado'][:300]}")
    console.print(
        f"\n[bold]{iguales}/{len(respuestas)}[/bold] answers identical word for word. "
        "Differences are fine: what matters is that the answers are still correct."
    )

    if out:
        lineas = [f"# Original vs quantized · {origen}", "",
                  f"Identical word for word: **{iguales}/{len(respuestas)}**", ""]
        for fila in respuestas:
            lineas += [f"### {fila['pregunta']}", "", "**Original**", "",
                       "```", fila["original"], "```", "", "**Quantized**", "",
                       "```", fila["cuantizado"], "```", ""]
        out.write_text("\n".join(lineas), encoding="utf-8")
        console.print(f"[green]Written[/green] -> {out}")


if __name__ == "__main__":
    app()
