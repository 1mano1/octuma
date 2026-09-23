import torch

from octuma.quantizer import QuantConfig, quantize_model

from .test_pipeline import VOCAB, calib, tiny_llama


def _logit_error(cfg):
    """Error de los logits contra el modelo sin cuantizar, misma semilla."""
    ref_model = tiny_llama()
    ids = torch.randint(0, VOCAB, (1, 32), generator=torch.Generator().manual_seed(7))
    ref = ref_model(ids).logits

    model = tiny_llama()
    quantize_model(model, calib(n=3), cfg)
    out = model(ids).logits
    return ((out - ref).norm() / ref.norm()).item()


def test_awq_pipeline_runs_and_keeps_model_usable():
    err = _logit_error(QuantConfig(bits=4, group_size=32, awq=True))
    assert err < 1.0
    assert err == err  # no NaN


def test_awq_does_not_break_shapes():
    model = tiny_llama()
    quantize_model(model, calib(n=2), QuantConfig(bits=4, group_size=32, awq=True))
    ids = torch.randint(0, VOCAB, (1, 16))
    assert model(ids).logits.shape == (1, 16, VOCAB)


def test_search_scale_flag_is_respected():
    model = tiny_llama()
    report = quantize_model(
        model, calib(n=2), QuantConfig(bits=4, group_size=32, search_scale=False)
    )
    assert len(report.layers) == 14


def test_awq_no_acumula_mas_muestras_de_las_que_usa(monkeypatch):
    """El recorte de AWQ tiene que ocurrir al guardar, no al concatenar.

    Antes el hook guardaba awq_samples filas de CADA lote y el torch.cat de
    despues se quedaba solo con las primeras awq_samples. Con los 128 lotes
    que usa `octuma quantize` por defecto, eso pedia 2.55 GB de una sola vez
    para el down_proj de un Qwen 0.5B y moria con "DefaultCPUAllocator: not
    enough memory" en una maquina de 30 GB. El resultado era identico: los
    127 lotes de mas se tiraban enteros.
    """
    import octuma.quantizer as Q

    model = tiny_llama()
    anchos = {
        m.in_features for m in model.modules() if isinstance(m, torch.nn.Linear)
    }
    cat_real = Q.torch.cat
    filas_juntadas = []

    def espia(tensores, *args, **kwargs):
        t = list(tensores)
        # las muestras de AWQ son 2D, float32 y en CPU, y su ancho es el
        # in_features de la capa; asi no se cuelan los cat internos de llama
        if (
            t
            and t[0].dim() == 2
            and t[0].dtype is torch.float32
            and not t[0].is_cuda
            and t[0].shape[1] in anchos
        ):
            filas_juntadas.append(sum(x.shape[0] for x in t))
        return cat_real(t, *args, **kwargs)

    monkeypatch.setattr(Q.torch, "cat", espia)

    tope = 8
    quantize_model(
        model,
        calib(n=6, seq=32),  # 6 lotes de 32 filas: 192 disponibles para un tope de 8
        QuantConfig(bits=4, group_size=32, awq=True, awq_samples=tope),
    )

    assert filas_juntadas, "no se vio ninguna concatenacion de muestras AWQ"
    assert max(filas_juntadas) <= tope, (
        f"se concatenaron {max(filas_juntadas)} filas para usar {tope}: el "
        "recorte volvio a quedarse al final del bucle"
    )
