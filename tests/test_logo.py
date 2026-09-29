"""El pulpo sale una vez por version y nunca en scripts ni en CI."""

from rich.console import Console

from octuma import __version__, logo


def _consola() -> Console:
    return Console(record=True, width=80, force_terminal=False)


def test_no_dibuja_fuera_de_una_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(logo, "_archivo_version", lambda: tmp_path / "v")
    monkeypatch.setattr(logo, "terminal_interactiva", lambda: False)
    c = _consola()
    logo.saludar_si_es_nueva(c)
    assert c.export_text() == ""
    # tampoco la marca como vista: el pulpo queda para la primera vez en terminal
    assert not (tmp_path / "v").exists()


def test_sale_una_sola_vez_por_version(monkeypatch, tmp_path):
    monkeypatch.setattr(logo, "_archivo_version", lambda: tmp_path / "v")
    monkeypatch.setattr(logo, "terminal_interactiva", lambda: True)
    c = _consola()
    logo.saludar_si_es_nueva(c)
    primera = c.export_text()
    assert "█" in primera and "installed" in primera

    logo.saludar_si_es_nueva(c)
    assert c.export_text() == "", "la segunda vez ya no se dibuja"


def test_avisa_la_actualizacion(monkeypatch, tmp_path):
    (tmp_path / "v").write_text("0.0.1", encoding="utf-8")
    monkeypatch.setattr(logo, "_archivo_version", lambda: tmp_path / "v")
    monkeypatch.setattr(logo, "terminal_interactiva", lambda: True)
    c = _consola()
    logo.saludar_si_es_nueva(c)
    assert f"0.0.1 -> {__version__}" in c.export_text()


def test_la_variable_de_entorno_lo_apaga(monkeypatch):
    monkeypatch.setenv("OCTUMA_NO_LOGO", "1")
    assert logo.terminal_interactiva() is False


def test_el_pulpo_es_simetrico():
    for fila in logo.PULPO.splitlines():
        fila = fila.ljust(logo.ANCHO_PULPO)
        assert fila == fila[::-1], fila


def test_el_nombre_va_al_lado_del_pulpo():
    c = Console(record=True, width=100, force_terminal=False)
    logo.dibujar(c, "0.1.3")
    filas = c.export_text().splitlines()
    assert "▄███▄ ▄████" in filas[logo.FILA_NOMBRE + 1], "el nombre al lado, no debajo"
    assert filas[logo.FILA_TEXTO].rstrip().endswith("0.1.3")


def test_en_terminal_angosta_no_se_enciman():
    c = Console(record=True, width=40, force_terminal=False)
    logo.dibujar(c, "0.1.3")
    assert all(len(f) <= 40 for f in c.export_text().splitlines())
