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


def test_version_se_puede_pedir_como_bandera():
    """`--version` es lo que se teclea sin pensar; existia solo `octuma version`.

    Se descubrio probando el paquete recien bajado de PyPI: el comando estaba
    ahi, pero la bandera universal contestaba "No such option".
    """
    from typer.testing import CliRunner

    from octuma import __version__

    r = CliRunner().invoke(cli.app, ["--version"])
    assert r.exit_code == 0, r.output
    assert __version__ in r.output

    # el subcomando de antes sigue funcionando: no se rompe a quien ya lo usaba
    r = CliRunner().invoke(cli.app, ["version"])
    assert r.exit_code == 0, r.output
    assert __version__ in r.output


def test_leer_meta_dice_que_la_carpeta_no_existe(tmp_path):
    """Sin esto, `info` y `compare` soltaban un FileNotFoundError crudo.

    Y era peor en `try` y `export`: transformers tomaba el nombre por un repo
    de Hugging Face y devolvia un 401 de veinte lineas hablando de tokens de
    autenticacion, cuando lo unico que pasaba es que `quantize` se habia
    cortado a la mitad sin escribir nada.
    """
    import pytest
    import typer

    with pytest.raises(typer.BadParameter) as e:
        cli._leer_meta(tmp_path / "no-existe")
    aviso = str(e.value)
    assert "no existe" in aviso
    assert "quantize" in aviso, "hay que decirle que revise el paso anterior"


def test_leer_meta_distingue_una_carpeta_que_no_es_tq(tmp_path):
    import pytest
    import typer

    with pytest.raises(typer.BadParameter) as e:
        cli._leer_meta(tmp_path)
    assert "octuma.json" in str(e.value)


def test_memoria_libre_usa_la_vram_libre_no_la_total(monkeypatch):
    """`total_memory` se imprimia con la palabra "libres".

    El aviso decia "la GPU tiene 8.6 GB libres" en una tarjeta de 8.6 GB
    totales, estuviera como estuviera de ocupada: siempre sonaba holgado.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1_000_000_000, 8_000_000_000))

    _, vram = cli._memoria_libre()
    assert vram == 1.0, "tiene que ser lo libre, no los 8 GB totales"


def test_memoria_libre_mira_tambien_la_ram_con_gpu_presente(monkeypatch):
    """Lo que mato al 0.5B fue la RAM, con la GPU medio vacia.

    Mirar solo la VRAM cuando hay CUDA dejaba el fallo real sin medir.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (8_000_000_000, 8_000_000_000))

    ram, vram = cli._memoria_libre()
    assert ram is not None and ram > 0
    assert vram == 8.0
