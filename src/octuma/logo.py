"""El pulpo de Octuma en la terminal.

pip no ejecuta codigo del paquete al instalar (un wheel solo se copia), asi que
el pulpo no puede salir durante `pip install`. Sale la primera vez que se usa
`octuma` despues de instalar o actualizar, y siempre con `octuma --version`.

Para no ensuciar la salida de scripts, CI o tuberias, solo se dibuja cuando la
salida es una terminal de verdad. `OCTUMA_NO_LOGO=1` lo apaga del todo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from . import __version__

# Sacado de assets/logo/octuma.png de la app, a 22 columnas.
PULPO = """\
       ▄██████▄
     ▄██████████▄
     ████████████
     ████████████
 ▄▄▄▄▀██ ████ ██▀▄▄▄▄
██▀██▄ ▀██████▀ ▄██▀██
▀▀ ███ ▄██████▄ ███ ▀▀
███ ▀████████████▀ ███
███▄  ▄▄██  ██▄▄  ▄███
 ▀█████▀▀    ▀▀█████▀"""

# "octuma" en letras de bloque, del mismo estilo que el logo de llama.cpp.
NOMBRE = """\
            ▄▄
▄███▄ ▄████ ██▄▄  ██ ██ ███▄███▄  ▀▀█▄
██ ██ ██    ██    ██ ██ ██ ██ ██ ▄█▀██
▀███▀ ▀████ ▀███  ▀████ ██ ██ ██ ▀█▄██"""

AZUL = "#3B7BFF"
ANCHO_PULPO = 22
# renglones del pulpo donde empiezan el nombre y el texto de abajo
FILA_NOMBRE = 2
FILA_TEXTO = 7


def _archivo_version() -> Path:
    """Donde se anota la ultima version que ya saludo."""
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"])
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "octuma" / "ultima_version"


def terminal_interactiva() -> bool:
    if os.environ.get("OCTUMA_NO_LOGO"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _banner(lineas: tuple[str, ...]) -> Text:
    """El pulpo a la izquierda y el nombre con las lineas a la derecha."""
    derecha: dict[int, tuple[str, str]] = {}
    for i, fila in enumerate(NOMBRE.splitlines()):
        derecha[FILA_NOMBRE + i] = (fila, "bold")
    for i, linea in enumerate(lineas):
        derecha[FILA_TEXTO + i] = (linea, "dim" if i else "")

    banner = Text()
    for i, fila in enumerate(PULPO.splitlines()):
        texto, estilo = derecha.get(i, ("", ""))
        if texto:
            banner.append(fila.ljust(ANCHO_PULPO), style=AZUL)
            banner.append("    ")
            banner.append(texto, style=estilo)
        else:
            banner.append(fila, style=AZUL)
        banner.append("\n")
    return banner


def dibujar(console: Console, *lineas: str) -> None:
    try:
        if console.width >= ANCHO_PULPO + 4 + max(len(f) for f in NOMBRE.splitlines()):
            console.print(_banner(lineas))
        else:
            # terminal angosta: el pulpo solo, y el texto debajo
            console.print(PULPO, style=AZUL, highlight=False)
            console.print("\n".join(("Octuma",) + lineas), highlight=False)
    except UnicodeEncodeError:
        # consola vieja sin los caracteres de bloque: mejor sin pulpo que un error
        console.print(" ".join(("Octuma",) + lineas))


def saludar_si_es_nueva(console: Console) -> None:
    """Dibuja el pulpo si esta version todavia no se habia usado en este equipo."""
    if not terminal_interactiva():
        return
    archivo = _archivo_version()
    try:
        anterior = archivo.read_text(encoding="utf-8").strip()
    except OSError:
        anterior = ""
    if anterior == __version__:
        return

    if anterior:
        dibujar(console, f"actualizado: {anterior} -> {__version__}")
    else:
        dibujar(console, f"{__version__} instalado", "Empieza con: octuma quantize <modelo>")
    anotar_version()


def anotar_version() -> None:
    """Marca esta version como ya saludada, para no repetir el pulpo."""
    archivo = _archivo_version()
    try:
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_text(__version__, encoding="utf-8")
    except OSError:
        # sin permiso para escribir: vuelve a salir la proxima vez, nada mas
        pass
