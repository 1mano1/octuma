"""La CLI de punta a punta, con un modelo diminuto y sin descargar nada.

`test_cli.py` fija los valores por defecto. Aqui se ejecutan los comandos de
verdad, en el orden en que los usa alguien que acaba de instalar: quantize,
info, evaluate, compare, try, analyze y export. Antes de este archivo la CLI
tenia un 21% de cobertura: los comandos existian, pero ninguna prueba los
corria, y un comando roto solo se habria notado en la maquina de un usuario.

El modelo es un Llama de dos bloques creado al vuelo, con un tokenizador BPE
entrenado sobre un texto de relleno. No sale nada a la red: donde un comando
bajaria wikitext-2, se le da ese mismo texto.
"""

from __future__ import annotations

import json

import pytest
import torch
from typer.testing import CliRunner

from octuma import cli

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

FRASES = [
    "the octopus has eight arms and three hearts",
    "a small model fits in a modest phone",
    "quantization keeps the weights in four bits",
    "the ocean is deep and the water is cold",
    "each group of weights shares one scale",
    "the phone answers without any network",
]
CORPUS = " . ".join(FRASES * 60)


@pytest.fixture(scope="module")
def origen(tmp_path_factory):
    """Carpeta con un modelo y un tokenizador diminutos, como un repo local."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    carpeta = tmp_path_factory.mktemp("origen")

    crudo = Tokenizer(models.BPE(unk_token="<unk>"))
    crudo.pre_tokenizer = pre_tokenizers.Whitespace()
    crudo.train_from_iterator(
        FRASES,
        trainers.BpeTrainer(
            vocab_size=160, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>"]
        ),
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=crudo,
        unk_token="<unk>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
    )
    # una plantilla minima: `try` la usa para armar la pregunta
    tok.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}"
    )
    tok.save_pretrained(carpeta)

    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(carpeta)

    (carpeta / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    return carpeta


@pytest.fixture(scope="module")
def cuantizado(origen, tmp_path_factory):
    """El resultado de `octuma quantize`, compartido por las demas pruebas."""
    salida = tmp_path_factory.mktemp("salida") / "tiny-int4"
    r = CliRunner().invoke(
        cli.app,
        [
            "quantize", str(origen),
            "--out", str(salida),
            "--calib", str(origen / "corpus.txt"),
            "--samples", "4", "--seqlen", "48",
            "--device", "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    return salida, r.output


@pytest.fixture()
def sin_red(monkeypatch, origen):
    """`evaluate` y `compare` miden con wikitext-2: aqui usan el corpus local."""
    from octuma import evaluate

    def ids_locales(tokenizer, split: str = "test"):
        return tokenizer(CORPUS, return_tensors="pt").input_ids

    monkeypatch.setattr(evaluate, "wikitext2_ids", ids_locales)


def test_quantize_escribe_una_carpeta_tq_completa(cuantizado):
    salida, texto = cuantizado
    assert "Done" in texto
    # lo que necesita cualquiera de los otros comandos para abrir la carpeta
    for archivo in ("octuma.json", "model.tq.safetensors", "config.json", "tokenizer.json"):
        assert (salida / archivo).exists(), f"falta {archivo}"

    meta = json.loads((salida / "octuma.json").read_text(encoding="utf-8"))
    assert len(meta["layers"]) == 14  # 7 lineales x 2 bloques
    assert all(capa["bits"] == 4 for capa in meta["layers"].values())
    # sin esto `compare` y `try --side-by-side` no saben contra que comparar
    assert meta["source_model"]
    assert meta["config"]["group_size"] == 32
    assert meta["config"]["awq"] is True


def test_quantize_sin_out_deduce_la_carpeta(origen, tmp_path, monkeypatch):
    """`octuma quantize <modelo>` a secas: la carpeta sale del nombre."""
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(
        cli.app,
        [
            "quantize", str(origen),
            "--calib", str(origen / "corpus.txt"),
            "--samples", "2", "--seqlen", "32",
            "--method", "rtn", "--no-awq", "--bits", "8",
            "--device", "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    esperado = tmp_path / f"{origen.name.lower()}-int8"
    assert (esperado / "octuma.json").exists()
    assert "Output folder" in r.output


def test_info_resume_el_modelo(cuantizado):
    salida, _ = cuantizado
    r = CliRunner().invoke(cli.app, ["info", str(salida)])
    assert r.exit_code == 0, r.output
    assert "INT4: 14 layers" in r.output
    assert "gptq" in r.output


def test_evaluate_mide_el_cuantizado_y_guarda_el_reporte(cuantizado, sin_red, tmp_path):
    salida, _ = cuantizado
    reporte = tmp_path / "eval.json"
    r = CliRunner().invoke(
        cli.app,
        ["evaluate", str(salida), "--windows", "3", "--seqlen", "32", "--speed",
         "--out", str(reporte)],
    )
    assert r.exit_code == 0, r.output
    datos = json.loads(reporte.read_text(encoding="utf-8"))
    assert datos["windows"] == 3
    assert datos["perplexity"] > 1.0
    assert datos["tokens_per_second"] > 0
    # un modelo cuantizado tiene que declarar pesos cuantizados
    assert datos["sizes"]["quantized"] > 0


def test_evaluate_tambien_mide_un_modelo_sin_cuantizar(origen, sin_red):
    r = CliRunner().invoke(
        cli.app, ["evaluate", str(origen), "--windows", "2", "--seqlen", "32"]
    )
    assert r.exit_code == 0, r.output
    assert "Perplexity" in r.output


def test_evaluate_rechaza_un_dataset_que_no_conoce(cuantizado):
    salida, _ = cuantizado
    r = CliRunner().invoke(cli.app, ["evaluate", str(salida), "--dataset", "inventado"])
    assert r.exit_code != 0
    assert "unsupported dataset" in r.output


def test_compare_pone_original_y_cuantizado_en_una_tabla(cuantizado, sin_red):
    salida, _ = cuantizado
    r = CliRunner().invoke(
        cli.app,
        ["compare", str(salida), "--windows", "3", "--seqlen", "32", "--device", "cpu"],
    )
    assert r.exit_code == 0, r.output
    assert "Original" in r.output and "Quantized" in r.output
    # la linea que resume: cuanto mas chico y a que costo
    assert "x smaller" in r.output and "perplexity" in r.output


def test_compare_sin_velocidad_no_pinta_esa_columna(cuantizado, sin_red):
    salida, _ = cuantizado
    r = CliRunner().invoke(
        cli.app,
        ["compare", str(salida), "--windows", "2", "--seqlen", "32", "--no-speed",
         "--device", "cpu"],
    )
    assert r.exit_code == 0, r.output
    assert "tok/s" not in r.output


def test_try_contesta_una_pregunta_y_sale(cuantizado):
    salida, _ = cuantizado
    r = CliRunner().invoke(
        cli.app,
        ["try", str(salida), "-p", "the octopus", "--max-new", "4", "--device", "cpu"],
    )
    assert r.exit_code == 0, r.output
    assert "the octopus" in r.output


def test_try_lado_a_lado_escribe_la_comparacion(cuantizado, tmp_path):
    salida, _ = cuantizado
    md = tmp_path / "lado.md"
    r = CliRunner().invoke(
        cli.app,
        ["try", str(salida), "--side-by-side", "--max-new", "3", "--device", "cpu",
         "--out", str(md)],
    )
    assert r.exit_code == 0, r.output
    assert f"/{len(cli.PREGUNTAS_PRUEBA)}" in r.output
    texto = md.read_text(encoding="utf-8")
    assert texto.count("**Original**") == len(cli.PREGUNTAS_PRUEBA)
    assert texto.count("**Quantized**") == len(cli.PREGUNTAS_PRUEBA)


def test_analyze_propone_un_plan_que_quantize_acepta(origen, tmp_path):
    """El plan de `analyze` tiene que servirle a `quantize --plan` tal cual."""
    plan = tmp_path / "plan.json"
    r = CliRunner().invoke(
        cli.app,
        [
            "analyze", str(origen),
            "--calib", str(origen / "corpus.txt"),
            "--samples", "2", "--seqlen", "32",
            "--group", "32", "--target-bits", "5.0",
            "--out", str(plan),
        ],
    )
    assert r.exit_code == 0, r.output
    capas = json.loads(plan.read_text(encoding="utf-8"))
    assert capas, "con 5 bits de promedio alguna capa tiene que subir a 8"
    assert set(capas.values()) == {8}

    salida = tmp_path / "mixto"
    r = CliRunner().invoke(
        cli.app,
        [
            "quantize", str(origen), "--out", str(salida),
            "--calib", str(origen / "corpus.txt"),
            "--samples", "2", "--seqlen", "32",
            "--plan", str(plan), "--device", "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    meta = json.loads((salida / "octuma.json").read_text(encoding="utf-8"))
    bits = {capa["bits"] for capa in meta["layers"].values()}
    assert bits == {4, 8}, "el plan no llego a las capas"
    a_ocho = sum(1 for capa in meta["layers"].values() if capa["bits"] == 8)
    assert a_ocho == len(capas)


def test_export_escribe_un_gguf_que_se_puede_leer(cuantizado, tmp_path):
    gguf = pytest.importorskip("gguf")
    salida, _ = cuantizado
    destino = tmp_path / "tiny.gguf"
    r = CliRunner().invoke(
        cli.app, ["export", str(salida), "--out", str(destino), "--name", "tiny"]
    )
    assert r.exit_code == 0, r.output

    lector = gguf.GGUFReader(str(destino))
    tipos = {t.name: t.tensor_type for t in lector.tensors}
    assert tipos["blk.0.attn_q.weight"] == gguf.GGMLQuantizationType.Q4_1
    # lo que anuncian llama.cpp y el visor de Hugging Face: estuvo en F16
    tipo = lector.fields["general.file_type"]
    assert int(tipo.parts[tipo.data[0]][0]) == int(gguf.LlamaFileType.MOSTLY_Q4_1)


def test_abre_una_carpeta_con_el_nombre_de_antes_del_renombrado(cuantizado, tmp_path):
    """Los modelos publicados en Hugging Face traen `tinyq.json`.

    El proyecto se llamaba TinyQ cuando se subieron. Tras el renombrado la CLI
    exigia `octuma.json` y contestaba "no es una carpeta cuantizada por Octuma"
    a sus propios tres modelos publicados.
    """
    import shutil

    salida, _ = cuantizado
    vieja = tmp_path / "publicado"
    shutil.copytree(salida, vieja)
    (vieja / "octuma.json").rename(vieja / "tinyq.json")

    r = CliRunner().invoke(cli.app, ["info", str(vieja)])
    assert r.exit_code == 0, r.output
    assert "INT4: 14 layers" in r.output

    r = CliRunner().invoke(
        cli.app, ["try", str(vieja), "-p", "the octopus", "--max-new", "3", "--device", "cpu"]
    )
    assert r.exit_code == 0, r.output


@pytest.mark.parametrize("comando", ["info", "compare", "try", "export"])
def test_una_carpeta_que_no_existe_se_explica(comando, tmp_path):
    """Los cuatro comandos que abren una carpeta .tq dicen lo mismo."""
    args = [comando, str(tmp_path / "no-existe")]
    if comando == "export":
        args += ["--out", str(tmp_path / "x.gguf")]
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code != 0
    assert "does not exist" in r.output


def test_ayuda_lista_todos_los_comandos():
    r = CliRunner().invoke(cli.app, ["--help"])
    assert r.exit_code == 0
    for comando in ("quantize", "compare", "try", "export", "info", "evaluate", "analyze"):
        assert comando in r.output
