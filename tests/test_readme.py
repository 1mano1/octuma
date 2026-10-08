"""Lo que el README enseña tiene que funcionar tal cual.

Un ejemplo de documentacion que deja de correr no avisa: se queda ahi,
escrito, hasta que alguien lo copia. Estas pruebas ejecutan el ejemplo de
"Uso desde Python" con un modelo diminuto en lugar del de Hugging Face.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from .test_pipeline import VOCAB, calib, tiny_llama

RAIZ = Path(__file__).resolve().parents[1]


def test_el_ejemplo_de_python_del_readme_funciona(tmp_path):
    from octuma.export.tq import load_quantized, save_quantized
    from octuma.quantizer import QuantConfig, quantize_model

    model = tiny_llama()
    report = quantize_model(model, calib(), QuantConfig())
    resumen = report.summary()
    assert resumen["n_layers"] == 14 and resumen["compression"] > 3.0

    save_quantized(model, tmp_path / "tiny-int4")
    cargado = load_quantized(tiny_llama(), tmp_path / "tiny-int4")

    ids = torch.randint(0, VOCAB, (1, 16))
    # las escalas se guardan en float16: de ahi la tolerancia
    assert torch.allclose(model(ids).logits, cargado(ids).logits, atol=2e-3)


def test_la_precision_mixta_del_readme_funciona():
    from octuma.quant.qlinear import QuantLinear
    from octuma.quantizer import QuantConfig, quantize_model

    model = tiny_llama()
    cfg = QuantConfig(bits=4, group_size=32, bits_overrides={"mlp.down_proj": 8})
    quantize_model(model, calib(), cfg)
    bits = {n: m.bits for n, m in model.named_modules() if isinstance(m, QuantLinear)}
    assert all(b == 8 for n, b in bits.items() if n.endswith("mlp.down_proj"))
    assert all(b == 4 for n, b in bits.items() if not n.endswith("mlp.down_proj"))


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md"])
def test_los_imports_del_readme_existen(archivo):
    """Cada `from octuma... import ...` de un bloque de codigo se puede importar."""
    import importlib

    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    encontrados = re.findall(r"^from (octuma[\w.]*) import ([\w, ]+)$", texto, re.MULTILINE)
    assert encontrados, "el README deberia enseñar al menos un import"
    for modulo, nombres in encontrados:
        mod = importlib.import_module(modulo)
        for nombre in (n.strip() for n in nombres.split(",")):
            assert hasattr(mod, nombre), f"{archivo}: {modulo} no tiene {nombre}"


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md"])
def test_los_comandos_del_readme_existen(archivo):
    """Cada `octuma <comando>` que se enseña es un comando de verdad."""
    from octuma import cli

    reales = {c.name or c.callback.__name__ for c in cli.app.registered_commands}
    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    usados = set(re.findall(r"^octuma ([a-z]+)\b", texto, re.MULTILINE))
    assert usados, "el README deberia enseñar al menos un comando"
    assert usados <= reales, f"{archivo} enseña comandos que no existen: {usados - reales}"


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md"])
def test_las_opciones_del_readme_existen(archivo):
    """Cada `--opcion` junto a un comando es una opcion de ese comando."""
    import typer

    from octuma import cli

    grupo = typer.main.get_command(cli.app)
    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    for linea in re.findall(r"^octuma [a-z]+ .*$", texto, re.MULTILINE):
        comando = linea.split()[1]
        sub = grupo.commands.get(comando)
        if sub is None:
            continue
        validas = {
            o
            for p in sub.params
            if getattr(p, "param_type_name", "") == "option"
            for o in list(p.opts) + list(p.secondary_opts)
        }
        assert validas, f"no se pudieron leer las opciones de `octuma {comando}`"
        for opcion in re.findall(r"(?<!\S)(--?[a-zA-Z][\w-]*)", linea.split("#")[0]):
            assert opcion in validas, f"{archivo}: `octuma {comando}` no tiene {opcion}"


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md"])
def test_los_archivos_que_enlaza_el_readme_existen(archivo):
    """Un enlace relativo roto solo se ve al hacer clic."""
    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    rutas = re.findall(r"\]\((?!https?://|#|mailto:)([^)#\s]+)", texto)
    rutas += re.findall(r'(?:src|href)="(?!https?://|#|mailto:)([^"#]+)"', texto)
    faltan = sorted({r for r in rutas if not (RAIZ / r).exists()})
    assert not faltan, f"{archivo} enlaza archivos que no existen: {faltan}"
