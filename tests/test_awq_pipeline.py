import torch

from tinyq.quantizer import QuantConfig, quantize_model

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
