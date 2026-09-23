"""Interfaz de linea de comandos de Octuma."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    add_completion=False,
    help="Octuma: cuantiza modelos de lenguaje a INT4/INT8 y los deja listos para equipos modestos y Android.",
)
console = Console()


def _mostrar_version(pedida: bool) -> None:
    if pedida:
        console.print(f"Octuma {__version__}")
        raise typer.Exit()


@app.callback()
def _principal(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_mostrar_version,
        is_eager=True,
        help="Muestra la version instalada y sale",
    ),
) -> None:
    """Punto de entrada: solo existe para colgar de el la bandera --version.

    El subcomando `octuma version` hace lo mismo y se queda por compatibilidad,
    pero lo que la gente teclea sin pensar es `--version`.
    """


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
        console.print("[yellow]No hay GPU disponible; se usa CPU[/yellow]")
        device, dtype = "cpu", "float32"
    if device == "cpu" and avisar_cpu:
        console.print(
            "[yellow]Sin GPU: en CPU esto va a tardar bastante.[/yellow] "
            "Con un modelo grande conviene una GPU, o prueba --bits 8."
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
            f"la carpeta '{model_dir}' no existe. Si acabas de cuantizar, "
            "revisa que 'octuma quantize' terminara: si se corto a la mitad "
            "no deja nada escrito."
        )
    if not (model_dir / "octuma.json").exists():
        raise typer.BadParameter(
            f"'{model_dir}' existe pero no tiene octuma.json, asi que no es "
            "una carpeta cuantizada por Octuma."
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
        partes.append(f"{ram:.1f} GB de RAM")
    if vram is not None:
        partes.append(f"{vram:.1f} GB de VRAM")
    if not partes:
        return
    console.print(
        f"Este modelo pide ~{necesario:.1f} GB y hay libres: " + " y ".join(partes)
    )

    corto = vram if en_gpu else ram
    if corto is not None and necesario > corto:
        console.print(
            "[yellow]Puede no caber.[/yellow] Opciones: --bits 8, un modelo mas "
            "chico, o liberar memoria."
        )
    # cuantizar con AWQ guarda muestras de activaciones en RAM aunque el modelo
    # este en la GPU. No depende del tamaño del modelo, asi que se avisa aparte.
    if ram is not None and ram < 4.0:
        console.print(
            f"[yellow]Quedan {ram:.1f} GB de RAM.[/yellow] Cuantizar con AWQ "
            "necesita RAM aunque el modelo corra en la GPU; si se queda corta, "
            "usa --no-awq o cierra algo."
        )


@app.command()
def version() -> None:
    """Muestra la version instalada."""
    console.print(f"Octuma {__version__}")


@app.command()
def quantize(
    model: str = typer.Argument(..., help="Id de Hugging Face o carpeta local"),
    out: Path = typer.Option(None, "--out", "-o", help="Carpeta de salida (por defecto se deduce del modelo)"),
    bits: int = typer.Option(4, help="Bits por peso: 2, 3, 4 u 8"),
    group_size: int = typer.Option(32, "--group", help="Pesos por grupo de escala"),
    method: str = typer.Option("gptq", help="gptq (con calibracion) o rtn (directo)"),
    calib: str = typer.Option("wikitext2", help="Dataset de calibracion o ruta a textos"),
    samples: int = typer.Option(128, help="Ventanas de calibracion"),
    seq_len: int = typer.Option(2048, "--seqlen", help="Tokens por ventana"),
    symmetric: bool = typer.Option(False, help="Cuantizacion simetrica"),
    awq: bool = typer.Option(True, help="Escalado AWQ antes de cuantizar"),
    search_scale: bool = typer.Option(True, help="Busca la escala optima por grupo"),
    device: str = typer.Option("auto", help="auto, cpu o cuda"),
    dtype: str = typer.Option("auto", help="auto, float16, bfloat16 o float32"),
    plan: Path = typer.Option(
        None, "--plan", help="JSON de precision mixta generado por 'octuma analyze'"
    ),
) -> None:
    """Calibra, cuantiza y guarda el modelo en formato .tq."""
    from .calibrate import load_calibration
    from .export.tq import disk_size, save_quantized
    from .quantizer import QuantConfig, quantize_model

    device, dtype = _resolver_device_dtype(device, dtype)
    if out is None:
        # sin --out: "Qwen/Qwen2.5-3B-Instruct" -> "qwen2.5-3b-instruct-int4"
        out = Path(f"{model.rstrip('/').split('/')[-1].lower()}-int{bits}")
        console.print(f"Carpeta de salida: [bold]{out}[/bold]")
    _avisar_memoria(model, device)

    console.print(f"[bold]Cargando[/bold] {model} ({dtype}, {device})")
    net, tok = _load_model(model, device, dtype)

    console.print(f"[bold]Calibrando[/bold] con {calib}: {samples} x {seq_len} tokens")
    cal = load_calibration(calib, tok, n_samples=samples, seq_len=seq_len)

    overrides: dict[str, int] = {}
    if plan:
        raw = json.loads(plan.read_text(encoding="utf-8"))
        overrides = {k.split(".", 2)[-1]: int(v) for k, v in raw.items()}
        console.print(f"Precision mixta: {len(overrides)} patrones a mas bits")

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
    console.print(f"[bold]Cuantizando[/bold] a INT{bits}, grupos de {group_size} ({extras})")
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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .evaluate import generation_speed, model_size_bytes, perplexity, wikitext2_ids
    from .export.tq import load_quantized

    tq_dir = Path(model)
    is_tq = (tq_dir / "octuma.json").exists()

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
def analyze(
    model: str = typer.Argument(..., help="Id de Hugging Face o carpeta local"),
    calib: str = typer.Option("wikitext2", help="Dataset de calibracion"),
    samples: int = typer.Option(16, help="Ventanas de calibracion"),
    seq_len: int = typer.Option(256, "--seqlen"),
    group_size: int = typer.Option(64, "--group"),
    target_bits: float = typer.Option(4.5, help="Promedio de bits objetivo"),
    device: str = typer.Option("cpu"),
    out: Path = typer.Option(None, "--out", "-o", help="Guarda el plan en JSON"),
) -> None:
    """Mide que capas sufren mas a 4 bits y propone un plan de precision mixta."""
    from .calibrate import load_calibration
    from .sensitivity import analyze_sensitivity

    console.print(f"[bold]Cargando[/bold] {model}")
    net, tok = _load_model(model, device)
    cal = load_calibration(calib, tok, n_samples=samples, seq_len=seq_len)

    console.print("[bold]Analizando sensibilidad[/bold] (error de salida por capa)")
    report = analyze_sensitivity(
        net, cal, bits_options=(4, 8), group_size=group_size,
        device=device, progress=console.print,
    )

    table = Table(title="Capas mas sensibles")
    table.add_column("Capa")
    table.add_column("Error INT4", justify="right")
    table.add_column("Error INT8", justify="right")
    for name, e4, e8 in report.table(top=12):
        table.add_row(name, f"{e4:.5f}", f"{e8:.5f}")
    console.print(table)

    plan = report.plan_mixed_precision(target_avg_bits=target_bits)
    console.print(f"Plan: {len(plan)} capas a INT8 para un promedio de {target_bits} bits")
    if out:
        out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        console.print(f"Plan -> {out}  (usalo con: octuma quantize ... --plan {out})")


@app.command()
def export(
    model_dir: Path = typer.Argument(..., help="Carpeta .tq generada por quantize"),
    out: Path = typer.Option(..., "--out", "-o", help="Archivo .gguf de salida"),
    name: str = typer.Option("octuma-model", help="Nombre dentro del GGUF"),
) -> None:
    """Convierte un modelo .tq a GGUF para llama.cpp y Android."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .export.gguf_export import export_gguf
    from .export.tq import load_quantized

    _leer_meta(model_dir)  # antes de que transformers lo tome por un repo del Hub
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
    meta = _leer_meta(model_dir)
    layers = meta["layers"]
    bits = {}
    for spec in layers.values():
        bits[spec["bits"]] = bits.get(spec["bits"], 0) + 1

    table = Table(title=f"Octuma · {model_dir.name}", show_header=False)
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


@app.command()
def compare(
    model_dir: Path = typer.Argument(..., help="Carpeta .tq generada por quantize"),
    original: str = typer.Option(None, help="Modelo sin cuantizar (por defecto, el que dice octuma.json)"),
    windows: int = typer.Option(20, help="Ventanas a evaluar"),
    seq_len: int = typer.Option(2048, "--seqlen"),
    device: str = typer.Option("auto", help="auto, cpu o cuda"),
    speed: bool = typer.Option(True, help="Mide tambien tokens por segundo"),
) -> None:
    """Compara el modelo cuantizado con el original: ¿quedo bien?"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .evaluate import generation_speed, model_size_bytes, perplexity, wikitext2_ids
    from .export.tq import load_quantized

    device, _ = _resolver_device_dtype(device, "auto", avisar_cpu=False)
    meta = _leer_meta(model_dir)
    original = original or meta.get("source_model")
    if not original:
        raise typer.BadParameter(
            "no se sabe de que modelo salio este .tq: pasa --original"
        )

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    ids = wikitext2_ids(tok)
    filas = []

    for etiqueta, cargar in (
        ("Original", lambda: AutoModelForCausalLM.from_pretrained(
            original, dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True).to(device).eval()),
        ("Cuantizado", lambda: load_quantized(
            AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_dir)),
            model_dir, device=device).to(device).eval()),
    ):
        console.print(f"[bold]Midiendo[/bold] {etiqueta.lower()}...")
        net = cargar()
        res = perplexity(net, ids, seq_len=seq_len, device=device,
                         dataset="wikitext2", max_windows=windows)
        mem = model_size_bytes(net)["total"] / 1e9
        tps = generation_speed(net, ids[:, :32], device=device) if speed else None
        filas.append((etiqueta, res.perplexity, mem, tps))
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = Table(title=f"{original} — {windows} ventanas de {seq_len}")
    table.add_column("")
    table.add_column("Perplejidad", justify="right")
    table.add_column("Memoria", justify="right")
    if speed:
        table.add_column("Velocidad", justify="right")
    for etiqueta, ppl, mem, tps in filas:
        fila = [etiqueta, f"{ppl:.3f}", f"{mem:.2f} GB"]
        if speed:
            fila.append(f"{tps:.1f} tok/s" if tps else "-")
        table.add_row(*fila)
    console.print(table)

    base, quant = filas[0], filas[1]
    perdida = (quant[1] / base[1] - 1) * 100
    console.print(
        f"[bold]{base[2] / quant[2]:.2f}x mas chico[/bold] por "
        f"[bold]{perdida:+.1f}%[/bold] de perplejidad"
    )


PREGUNTAS_PRUEBA = [
    "Explica en dos frases que es la cuantizacion de modelos.",
    "¿Cual es la capital de Australia?",
    "Escribe una funcion de Python que invierta una cadena.",
    "¿Cuanto es 17 por 24? Muestra el procedimiento.",
    "Traduce al ingles: 'El gato duerme en la ventana'.",
    "¿Que es mas pesado, un kilo de plomo o un kilo de plumas?",
    "Escribe un haiku sobre la lluvia.",
    "¿En que año llego el hombre a la Luna?",
    "Explica la diferencia entre una lista y una tupla en Python.",
    "Termina el refran: 'Mas vale pajaro en mano...'",
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
    model_dir: Path = typer.Argument(..., help="Carpeta .tq generada por quantize"),
    side_by_side: bool = typer.Option(
        False, "--side-by-side", help="Compara las respuestas con las del original"
    ),
    prompt: str = typer.Option(None, "-p", help="Una sola pregunta y salir"),
    max_new: int = typer.Option(120, help="Tokens por respuesta"),
    device: str = typer.Option("auto", help="auto, cpu o cuda"),
    out: Path = typer.Option(None, "--out", "-o", help="Guarda la comparacion en Markdown"),
) -> None:
    """Prueba el modelo cuantizado: ¿sigue hablando bien?

    La perplejidad no contesta esto. Ver las respuestas si.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .export.tq import load_quantized

    device, _ = _resolver_device_dtype(device, "auto", avisar_cpu=False)
    _leer_meta(model_dir)  # antes de que transformers lo tome por un repo del Hub
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    console.print("[bold]Cargando[/bold] el modelo cuantizado...")
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
        console.print("Escribe tu pregunta ([dim]Ctrl+C para salir[/dim])\n")
        while True:
            try:
                pregunta = typer.prompt(">")
            except (KeyboardInterrupt, EOFError):
                console.print("\nHasta luego")
                return
            console.print(f"\n{_responder(net, tok, pregunta, max_new, device)}\n")

    meta = json.loads((model_dir / "octuma.json").read_text(encoding="utf-8"))
    origen = meta.get("source_model")
    if not origen:
        raise typer.BadParameter("el .tq no dice de que modelo salio")

    respuestas = []
    for p in PREGUNTAS_PRUEBA:
        respuestas.append({"pregunta": p, "cuantizado": _responder(net, tok, p, max_new, device)})
        console.print(f"  [dim]cuantizado:[/dim] {p[:45]}...")
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print("[bold]Cargando[/bold] el original para comparar...")
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
        console.print(f"[dim]cuantizado[/dim] {fila['cuantizado'][:300]}")
    console.print(
        f"\n[bold]{iguales}/{len(respuestas)}[/bold] respuestas identicas palabra por palabra. "
        "Que difieran no es malo: lo que importa es que sigan siendo correctas."
    )

    if out:
        lineas = [f"# Original vs cuantizado — {origen}", "",
                  f"Identicas palabra por palabra: **{iguales}/{len(respuestas)}**", ""]
        for fila in respuestas:
            lineas += [f"### {fila['pregunta']}", "", "**Original**", "",
                       "```", fila["original"], "```", "", "**Cuantizado**", "",
                       "```", fila["cuantizado"], "```", ""]
        out.write_text("\n".join(lineas), encoding="utf-8")
        console.print(f"[green]Escrito[/green] -> {out}")


if __name__ == "__main__":
    app()
