"""La calibracion es la entrada de todo lo demas: si sale mal, nada lo delata."""

from __future__ import annotations

import pytest
import torch

from octuma.calibrate import build_from_texts, load_calibration


class _Tok:
    """Tokenizador de juguete: una palabra, un id."""

    def __call__(self, texto: str, return_tensors: str = "pt"):
        ids = [hash(p) % 97 for p in texto.split()]

        class _Salida:
            input_ids = torch.tensor([ids])

        return _Salida()


TEXTO = " ".join(f"palabra{i}" for i in range(400))


def test_las_ventanas_tienen_el_tamano_pedido():
    cal = build_from_texts([TEXTO], _Tok(), n_samples=6, seq_len=32)
    assert len(cal) == 6
    assert all(lote.shape == (1, 32) for lote in cal.batches)
    assert cal.n_tokens == 6 * 32
    assert cal.seq_len == 32


def test_la_misma_semilla_da_las_mismas_ventanas():
    """Sin esto dos corridas "iguales" no serian comparables."""
    a = build_from_texts([TEXTO], _Tok(), n_samples=4, seq_len=16, seed=7)
    b = build_from_texts([TEXTO], _Tok(), n_samples=4, seq_len=16, seed=7)
    c = build_from_texts([TEXTO], _Tok(), n_samples=4, seq_len=16, seed=8)
    assert all(torch.equal(x, y) for x, y in zip(a.batches, b.batches, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(a.batches, c.batches, strict=True))


def test_los_lotes_agrupan_las_ventanas():
    cal = build_from_texts([TEXTO], _Tok(), n_samples=5, seq_len=16, batch_size=2)
    assert [lote.shape[0] for lote in cal.batches] == [2, 2, 1]


def test_un_corpus_mas_corto_que_una_ventana_se_explica():
    """Mejor un error claro que ventanas rellenas de nada."""
    with pytest.raises(ValueError, match="at least 65 are needed"):
        build_from_texts(["solo tres palabras"], _Tok(), n_samples=2, seq_len=64)


def test_los_textos_vacios_no_cuentan():
    con = build_from_texts(["", "   ", TEXTO], _Tok(), n_samples=2, seq_len=16)
    sin = build_from_texts([TEXTO], _Tok(), n_samples=2, seq_len=16)
    assert all(torch.equal(x, y) for x, y in zip(con.batches, sin.batches, strict=True))


def test_calibrar_con_un_archivo_propio(tmp_path):
    archivo = tmp_path / "mis-textos.txt"
    archivo.write_text(TEXTO, encoding="utf-8")
    cal = load_calibration(str(archivo), _Tok(), n_samples=3, seq_len=16)
    assert len(cal) == 3
    # queda anotado de donde salio: va al octuma.json del modelo
    assert cal.source == str(archivo)


def test_calibrar_con_una_carpeta_junta_todos_los_txt(tmp_path):
    (tmp_path / "a.txt").write_text(TEXTO[: len(TEXTO) // 2], encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text(TEXTO[len(TEXTO) // 2 :], encoding="utf-8")
    (tmp_path / "ignorado.md").write_text("no es txt " * 500, encoding="utf-8")

    cal = load_calibration(str(tmp_path), _Tok(), n_samples=2, seq_len=300)
    # 300 tokens solo caben juntando los dos .txt; cada uno tiene ~200
    assert len(cal) == 2


def test_un_dataset_que_no_existe_no_cae_a_un_default():
    with pytest.raises(ValueError, match="unknown calibration dataset"):
        load_calibration("no-existe-ni-es-ruta", _Tok())
