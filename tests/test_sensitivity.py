import torch

from tinyq.quant.core import quantize_tensor
from tinyq.sensitivity import (
    SensitivityReport,
    LayerSensitivity,
    analyze_sensitivity,
    output_error,
)

from .test_pipeline import calib, tiny_llama


def test_output_error_grows_when_bits_drop():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    x = torch.randn(256, 64)
    H = x.T @ x
    e4 = output_error(w, quantize_tensor(w, 4, 32).dequantize(), H)
    e8 = output_error(w, quantize_tensor(w, 8, 32).dequantize(), H)
    assert e4 > e8 > 0


def test_output_error_is_zero_for_identical_weights():
    w = torch.randn(8, 16)
    H = torch.eye(16)
    assert output_error(w, w.clone(), H) < 1e-12


def test_analyze_sensitivity_covers_all_linears():
    model = tiny_llama()
    report = analyze_sensitivity(model, calib(), bits_options=(4, 8), group_size=32)
    assert len(report.layers) == 14
    assert all(l.error_by_bits[4] >= l.error_by_bits[8] for l in report.layers)


def test_plan_respects_budget():
    layers = [
        LayerSensitivity(f"blocks.0.l{i}", n_params=1000, error_by_bits={4: 0.1 * (10 - i), 8: 0.0})
        for i in range(10)
    ]
    report = SensitivityReport(layers=layers)
    plan = report.plan_mixed_precision(target_avg_bits=4.4, high_bits=8, low_bits=4)
    # presupuesto: 0.4/4 = 10% de los parametros a 8 bits => 1 capa de 10
    assert len(plan) == 1
    assert plan == {"blocks.0.l0": 8}


def test_plan_picks_most_sensitive_first():
    layers = [
        LayerSensitivity("a", 100, {4: 0.01, 8: 0.001}),
        LayerSensitivity("b", 100, {4: 0.50, 8: 0.002}),
    ]
    plan = SensitivityReport(layers=layers).plan_mixed_precision(target_avg_bits=6.0)
    assert "b" in plan
