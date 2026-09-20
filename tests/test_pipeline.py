import torch

from tinyq.calibrate import CalibrationSet
from tinyq.evaluate import model_size_bytes, perplexity
from tinyq.export.tq import load_quantized, save_quantized
from tinyq.quant.qlinear import QuantLinear
from tinyq.quantizer import QuantConfig, find_blocks, quantize_model

VOCAB = 128


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def calib(n=2, seq=32):
    torch.manual_seed(1)
    return CalibrationSet(
        batches=[torch.randint(0, VOCAB, (1, seq)) for _ in range(n)],
        seq_len=seq,
        source="random",
    )


def test_find_blocks_on_llama():
    blocks, path = find_blocks(tiny_llama())
    assert path == "model.layers"
    assert len(blocks) == 2


def test_quantize_model_replaces_linears_and_reports():
    model = tiny_llama()
    report = quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))

    qlayers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    assert len(qlayers) == len(report.layers) == 14  # 7 lineales x 2 bloques
    assert report.compression > 3.0
    assert all(0.0 < l.rel_fro < 0.5 for l in report.layers)
    # lm_head se queda en punto flotante
    assert not isinstance(model.lm_head, QuantLinear)


def test_quantized_model_still_produces_finite_logits():
    model = tiny_llama()
    ids = torch.randint(0, VOCAB, (1, 32))
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    out = model(ids).logits
    assert out.shape == (1, 32, VOCAB)
    assert torch.isfinite(out).all()


def test_rtn_method_runs():
    model = tiny_llama()
    report = quantize_model(model, calib(), QuantConfig(method="rtn", bits=8, group_size=32))
    assert report.q_bytes < report.fp_bytes


def test_bits_overrides_apply():
    model = tiny_llama()
    cfg = QuantConfig(bits=4, group_size=32, bits_overrides={"mlp.down_proj": 8})
    report = quantize_model(model, calib(), cfg)
    by_name = {l.name: l.bits for l in report.layers}
    assert by_name["blocks.0.mlp.down_proj"] == 8
    assert by_name["blocks.0.self_attn.q_proj"] == 4


def test_save_and_load_roundtrip(tmp_path):
    model = tiny_llama()
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    ids = torch.randint(0, VOCAB, (1, 32))
    before = model(ids).logits

    save_quantized(model, tmp_path, cfg=QuantConfig(bits=4, group_size=32), fp16_dense=False)
    restored = load_quantized(tiny_llama(), tmp_path)
    after = restored(ids).logits

    assert torch.allclose(before, after, atol=1e-4)


def test_fp16_dense_shrinks_file_without_breaking_outputs(tmp_path):
    model = tiny_llama()
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    ids = torch.randint(0, VOCAB, (1, 32))
    before = model(ids).logits

    save_quantized(model, tmp_path / "fp32", fp16_dense=False)
    save_quantized(model, tmp_path / "fp16", fp16_dense=True)
    from tinyq.export.tq import disk_size

    assert disk_size(tmp_path / "fp16") < disk_size(tmp_path / "fp32")

    after = load_quantized(tiny_llama(), tmp_path / "fp16")(ids).logits
    rel = (after - before).norm() / before.norm()
    assert rel < 5e-3


def test_save_and_load_with_tied_embeddings(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    ids = torch.randint(0, VOCAB, (1, 32))
    before = model(ids).logits

    save_quantized(model, tmp_path, fp16_dense=False)
    restored = LlamaForCausalLM(cfg).eval()
    restored = load_quantized(restored, tmp_path)
    assert torch.allclose(before, restored(ids).logits, atol=1e-4)


def test_model_size_reports_savings():
    model = tiny_llama()
    fp = model_size_bytes(model)
    quantize_model(model, calib(), QuantConfig(bits=4, group_size=32))
    q = model_size_bytes(model)
    assert q["quantized"] > 0
    assert q["total"] < fp["total"]


def test_perplexity_is_finite():
    model = tiny_llama()
    ids = torch.randint(0, VOCAB, (1, 64))
    res = perplexity(model, ids, seq_len=32, dataset="random")
    assert res.n_windows == 2
    assert res.perplexity > 1.0
