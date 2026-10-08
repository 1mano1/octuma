"""Los documentos del repositorio no se pueden desfasar entre si.

`test_readme.py` comprueba que lo que el README enseña existe en el codigo.
Aqui se comprueba el propio documento: que sus enlaces lleven a algo, que el
español y el ingles digan lo mismo y que la cabecera animada traiga las cifras
medidas.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/1mano1/octuma/main/"
BLOB = "https://github.com/1mano1/octuma/blob/main/"
TREE = "https://github.com/1mano1/octuma/tree/main/"


def _leer(archivo: str) -> str:
    return (RAIZ / archivo).read_text(encoding="utf-8")


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md", "llms.txt"])
def test_los_enlaces_a_este_repo_apuntan_a_archivos_que_existen(archivo):
    """Las imagenes y varios enlaces van con la direccion completa de GitHub.

    Es para que tambien se vean en PyPI, que no resuelve rutas relativas. La
    contra es que un nombre mal escrito ya no lo delata nada: por eso se
    comprueba aqui que cada uno existe en el repositorio.
    """
    texto = _leer(archivo)
    rutas: set[str] = set()
    for base in (RAW, BLOB, TREE):
        rutas |= set(re.findall(re.escape(base) + r"([\w./-]+)", texto))
    assert rutas, f"{archivo} deberia enlazar algo del repositorio"
    faltan = sorted(r for r in rutas if not (RAIZ / r.rstrip(".")).exists())
    assert not faltan, f"{archivo} enlaza archivos que no existen: {faltan}"


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md"])
def test_no_hay_enlaces_relativos(archivo):
    """El README en ingles es la pagina de PyPI, y PyPI no resuelve rutas
    relativas: un enlace a `runs/...` sale roto ahi, y una version publicada
    ya no se puede corregir."""
    texto = _leer(archivo)
    relativos = re.findall(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)", texto)
    relativos += re.findall(r'(?:src|href)="(?!https?://|#|mailto:)([^"]+)"', texto)
    assert not relativos, f"{archivo} tiene enlaces relativos: {relativos}"


@pytest.mark.parametrize("archivo", ["README.md", "README.en.md"])
def test_los_enlaces_internos_llevan_a_un_titulo(archivo):
    """Cada `#ancla` del indice tiene que corresponder a un titulo."""
    texto = _leer(archivo)

    def ancla(titulo: str) -> str:
        # como las arma GitHub: minusculas, sin signos, espacios a guiones
        limpio = re.sub(r"[^\w\- ]", "", titulo.strip().lower().replace("`", ""))
        return limpio.replace(" ", "-")

    titulos = {ancla(t) for t in re.findall(r"^#{1,6} (.+)$", texto, re.MULTILINE)}
    usadas = set(re.findall(r'(?:\]\(|href=")#([^)"]+)', texto))
    assert usadas, "el README deberia tener un indice"
    assert usadas <= titulos, f"{archivo}: anclas sin titulo: {sorted(usadas - titulos)}"


def test_los_dos_readme_tienen_las_mismas_secciones():
    """El español y el ingles son el mismo documento.

    Si uno gana una seccion y el otro no, alguien esta leyendo una version
    vieja sin saberlo.
    """

    def forma(archivo: str) -> list[int]:
        return [len(m) for m in re.findall(r"^(#{1,6}) ", _leer(archivo), re.MULTILINE)]

    assert forma("README.md") == forma("README.en.md")


def test_los_dos_readme_dan_los_mismos_numeros():
    """Las cifras no se traducen: tienen que ser las mismas en los dos."""

    def cifras(archivo: str) -> list[str]:
        return sorted(re.findall(r"\d+\.\d+(?:%|x| GB)", _leer(archivo)))

    assert cifras("README.md") == cifras("README.en.md")


def test_el_readme_dice_cuantas_pruebas_hay(request):
    """"N pruebas" en la seccion de desarrollo tiene que ser el numero real."""
    reales = len(request.session.items)
    if reales < 50:
        pytest.skip("solo se puede comprobar corriendo la bateria completa")
    for archivo, patron in (("README.md", r"# (\d+) pruebas"), ("README.en.md", r"# (\d+) tests")):
        dicho = int(re.search(patron, _leer(archivo)).group(1))
        assert dicho == reales, f"{archivo} dice {dicho} y hay {reales}"


def test_la_cabecera_animada_esta_al_dia_con_las_mediciones():
    """`hero*.svg` se genera desde runs/: si la medicion cambia, se regenera."""
    spec = importlib.util.spec_from_file_location("hero_svg", RAIZ / "scripts" / "hero_svg.py")
    hero = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hero)
    m = hero.medicion()
    for nombre in ("hero.svg", "hero-es.svg"):
        svg = _leer(f"docs/img/{nombre}")
        for dato in (f"{m['antes']:.2f} GB", f"{m['despues']:.2f} GB", f"+{m['perdida']:.2f}%"):
            assert dato in svg, f"{nombre} no trae {dato}: corre scripts/hero_svg.py"


def test_las_tablas_del_readme_salen_de_las_mediciones():
    """Las cifras de la tabla de llama.cpp son las de runs/gguf__*.json."""
    import json

    texto = _leer("README.md")
    for slug in ("qwen0.5b", "qwen1.5b", "qwen3b"):
        datos = json.loads(_leer(f"runs/gguf__{slug}.json"))
        for r in datos["resultados"]:
            if r["familia"] == "original":
                continue
            celda = f"+{r['dano_pct']:.2f}%"
            assert celda in texto, f"{slug} / {r['etiqueta']}: falta {celda} en el README"
            assert f"{r['bytes'] / 1e9:.2f} GB" in texto, f"{slug} / {r['etiqueta']}: tamaño"


def test_la_version_del_changelog_es_la_del_paquete():
    from octuma import __version__

    primera = re.search(r"^## (\d+\.\d+\.\d+)", _leer("CHANGELOG.md"), re.MULTILINE).group(1)
    assert primera == __version__, "la entrada mas nueva del CHANGELOG no es la version actual"


def test_las_imagenes_del_repo_no_estan_danadas():
    """Una PNG empieza con 8 bytes fijos que incluyen un salto de linea CRLF.

    Con `core.autocrlf` en Windows, git trato las graficas como texto y
    convirtio ese salto: las doce quedaron corruptas desde septiembre y GitHub
    las mostraba como un hueco. `.gitattributes` las marca ahora como binarias.
    """
    firma = b"\x89PNG\r\n\x1a\n"
    rotas = [
        str(p.relative_to(RAIZ))
        for p in (RAIZ / "docs" / "img").glob("*.png")
        if p.read_bytes()[:8] != firma
    ]
    assert not rotas, f"imagenes dañadas: {rotas}"
