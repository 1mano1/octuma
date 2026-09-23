"""La CLI es lo primero que toca quien instala Octuma: sus defaults importan."""

import inspect

from octuma import cli


def _defaults(fn) -> dict:
    """Valor por defecto de cada opcion de un comando de typer."""
    out = {}
    for nombre, par in inspect.signature(fn).parameters.items():
        d = par.default
        out[nombre] = getattr(d, "default", d)
    return out


def test_quantize_usa_la_configuracion_ganadora():
    """Los defaults deben ser lo que gana en el barrido, no algo peor.

    Durante meses `octuma quantize` venia con AWQ apagado y calibracion 64x512,
    mientras la tabla del README se midio con AWQ y 128x2048: quien corria el
    comando obvio sacaba resultados peores que los publicados.
    """
    d = _defaults(cli.quantize)
    assert d["awq"] is True, "GPTQ+AWQ gana en los cuatro modelos medidos"
    assert d["method"] == "gptq"
    assert d["samples"] == 128
    assert d["seq_len"] == 2048
    # el exportador a GGUF exige grupos de 32
    assert d["group_size"] == 32


def test_quantize_no_obliga_a_elegir_hardware_ni_carpeta():
    d = _defaults(cli.quantize)
    assert d["device"] == "auto"
    assert d["dtype"] == "auto"
    assert d["out"] is None, "sin --out la carpeta se deduce del nombre del modelo"


def test_device_auto_cae_a_cpu_sin_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, dtype = cli._resolver_device_dtype("auto", "auto", avisar_cpu=False)
    assert device == "cpu"
    # float16 en CPU va lentisimo o ni siquiera esta soportado
    assert dtype == "float32"


def test_device_cuda_pedido_sin_gpu_no_revienta(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, dtype = cli._resolver_device_dtype("cuda", "auto", avisar_cpu=False)
    assert device == "cpu" and dtype == "float32"


def test_existen_los_tres_comandos_de_entrada():
    """quantize, compare y try son los que ve alguien que acaba de instalar."""
    nombres = {c.name or c.callback.__name__ for c in cli.app.registered_commands}
    assert {"quantize", "compare", "try"} <= nombres
