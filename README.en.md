# Octuma

[Español](https://github.com/1mano1/octuma/blob/main/README.md) · **English**

INT4/INT8 quantization for language models, built so they run on modest
laptops, small servers and Android phones.

A 3B model in FP16 needs ~6.2 GB of memory. Octuma brings it down to **2.4 GB
while losing 2.4% of quality**, and the resulting file runs in llama.cpp, which
is what any phone or laptop without a GPU uses.

```bash
pip install "octuma[hf,gguf]"

octuma quantize Qwen/Qwen2.5-3B-Instruct    # quantize, no choices needed
octuma compare qwen2.5-3b-instruct-int4     # did it come out well?
octuma try qwen2.5-3b-instruct-int4         # does it still talk well?
```

You don't pick a method, bit width or folder: the defaults are the setup that
won our own measurements, the GPU is detected automatically, and it warns you
**before downloading anything** if the model won't fit in your memory.

> **The project used to be called TinyQ** and was renamed Octuma on
> 2026-09-22, to share its name with the Android app that runs these models.
> What you install, what you import and the terminal command are all
> `octuma`. Old links on GitHub and Hugging Face redirect on their own.

## Installation

```bash
# the usual: quantize Hugging Face models and export to GGUF
pip install "octuma[hf,gguf]"

# the development version, before it reaches PyPI
pip install "octuma[hf,gguf] @ git+https://github.com/1mano1/octuma.git"
```

Needs Python 3.10+ and PyTorch.

**Install the extras.** A bare `pip install octuma` leaves out `transformers`,
and without it only `octuma --version` and `octuma info` work: you can't
quantize, compare or export. pip calls them optional, but you can't skip them.

| Extra | What for | Without it |
|---|---|---|
| `hf` | `transformers` and `datasets` | no `quantize`, `compare`, `try` or `export` |
| `gguf` | Export to GGUF for llama.cpp and Android | no `export` |
| `dev` | `pytest` and `ruff`, for development | — |

### CPU only

```bash
pip install "octuma[hf,gguf]"
```

This is the standard install. On Windows pip brings the CPU build of PyTorch;
on Linux and macOS it also works without a GPU.

### With an NVIDIA GPU

PyTorch with CUDA is not on PyPI: it has to be installed first from its own
index. If this step is skipped, `octuma quantize` runs on the CPU without a
warning.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "octuma[hf,gguf]"     # keeps the torch already installed
```

`cu126` has PyTorch for Python 3.10 to 3.14. The error `No matching
distribution found for torch` almost always means the index has no build for
the installed Python version; `cu121`, for example, stops at 3.12.

## Try it on your PC

Everything stays inside one folder that is deleted at the end. The commands
are for PowerShell; on Linux or macOS activate the environment with
`source .venv/bin/activate`. There is no need to download a model first:
`quantize` fetches Qwen2.5 0.5B from Hugging Face (~1 GB, no account needed).

### CPU only

Calibration and comparison are shortened so they finish in minutes instead of
hours.

```powershell
mkdir C:\octuma-test
cd C:\octuma-test
Set-ExecutionPolicy -Scope Process Bypass   # allows activating the environment, this terminal only
py -m venv .venv                            # with several versions: py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install "octuma[hf,gguf]"
octuma --version

octuma quantize Qwen/Qwen2.5-0.5B-Instruct --samples 32 --seqlen 512 --out qwen05b
octuma compare qwen05b --windows 5 --seqlen 512
octuma try qwen05b -p "What is the capital of Australia?"
octuma export qwen05b --out qwen05b.gguf
```

`compare` spends several minutes without printing anything while it measures;
it is not stuck.

### With an NVIDIA GPU

With the full setup, the same one used in the tables below.

```powershell
mkdir C:\octuma-test
cd C:\octuma-test
Set-ExecutionPolicy -Scope Process Bypass
py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "octuma[hf,gguf]"
python -c "import torch; print(torch.cuda.is_available())"   # should print True
octuma --version

octuma quantize Qwen/Qwen2.5-0.5B-Instruct --out qwen05b
octuma compare qwen05b
octuma try qwen05b -p "What is the capital of Australia?"
octuma export qwen05b --out qwen05b.gguf
```

The CPU test numbers can't be compared with the tables, which use 128 windows
of 2048 tokens; the GPU ones can.

### Cleaning up

```powershell
deactivate
cd C:\
Remove-Item -Recurse -Force C:\octuma-test
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
Remove-Item -Recurse -Force "$env:APPDATA\octuma"   # record of the last version used
```

The wikitext data stays in `.cache\huggingface\hub\datasets--*`.

## Ready-to-use models

Three Qwen2.5 models already quantized, each with the `.tq` folder (PyTorch)
and the `.gguf` (llama.cpp / Android):

| Model | INT4 size | Perplexity | vs original | Original's license |
|---|---|---|---|---|
| [qwen0.5b-int4-Octuma](https://huggingface.co/Imanol11/qwen0.5b-int4-Octuma) | 0.52 GB | 12.710 | +3.73% | Apache 2.0 |
| [qwen1.5b-int4-Octuma](https://huggingface.co/Imanol11/qwen1.5b-int4-Octuma) | 1.32 GB | 8.486 | +2.11% | Apache 2.0 |
| [qwen3b-int4-Octuma](https://huggingface.co/Imanol11/qwen3b-int4-Octuma) | 2.40 GB | 7.512 | +2.43% | **Qwen Research (non-commercial)** |

> **The 3B can't be used commercially.** Unlike the rest of the family,
> [`Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
> isn't Apache 2.0 but the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE):
> research and evaluation only. The quantized model inherits that condition.
> If you want something commercial, the 1.5B and 0.5B are Apache 2.0, and so
> is the [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

```bash
hf download Imanol11/qwen3b-int4-Octuma --local-dir qwen3b
llama-cli -m qwen3b/qwen3b-int4.gguf -p "Hello"
```

## How good is it

### Against llama.cpp's formats, inside llama.cpp

This is the comparison that matters for running locally: Octuma's `.gguf`
measured with the same engine, the same corpus and the same windows as its
rivals.

![Degradation by model size](docs/img/gguf-por-tamano.png#gh-light-mode-only)
![Degradation by model size](docs/img/gguf-por-tamano-dark.png#gh-dark-mode-only)

| Model | Octuma INT4 | Q4_K_M | Q4_0 |
|---|---|---|---|
| Qwen2.5-0.5B | +3.73% | **+2.63%** | +13.17% |
| Qwen2.5-1.5B | **+2.11%** | +4.74% | +8.35% |
| Qwen2.5-3B | **+2.43%** | +6.21% | +11.02% |

Quality lost against the same original in F16; lower is better, and the best
in each row is in bold.

**As the model grows, Q4_K_M keeps getting worse (2.63% → 4.74% → 6.21%)
while Octuma stays flat (3.73% → 2.11% → 2.43%).** On the 3B, Octuma does
**2.6 times less damage** than Q4_K_M, the most common format for running
models locally.

**On the 0.5B, Q4_K_M wins**: it loses less and takes less space (0.40 GB vs
0.52). The crossover is between 0.5B and 1.5B. If your model is tiny, use
Q4_K_M; Octuma pays off as the model grows, which is exactly when memory gets
tight.

Per-model detail, with charts and the full table, in
[`runs/COMPARATIVA_GGUF.md`](runs/COMPARATIVA_GGUF.md) (in Spanish).

### Against bitsandbytes, inside PyTorch

![Comparison against other quantizers](docs/img/comparativa-herramientas.png#gh-light-mode-only)
![Comparison against other quantizers](docs/img/comparativa-herramientas-dark.png#gh-dark-mode-only)

On the same Qwen2.5-3B, with the same evaluator and the same windows, Octuma
does **less than half the damage** of bitsandbytes NF4, Hugging Face's default
quantizer and the one QLoRA uses:

| Tool | Perplexity | Memory | Loss |
|---|---|---|---|
| FP16 (not quantized) | 8.347 | 6.79 GB | — |
| **Octuma GPTQ+AWQ** | **8.549** | 2.76 GB | **+2.4%** |
| Octuma GPTQ | 8.578 | 2.76 GB | +2.8% |
| bitsandbytes NF4 | 8.906 | 2.63 GB | +6.7% |
| bitsandbytes FP4 | 13.343 | 2.63 GB | +59.9% |

> The perplexities in this table and the previous one **can't be compared with
> each other**: each engine splits and averages differently, and the same F16
> gives 8.347 here and 7.334 in llama.cpp. What's comparable is always the
> loss within one engine.

And the damage goes down as the model grows, in PyTorch too:

![Degradation by model size](docs/img/degradacion-por-tamano.png#gh-light-mode-only)
![Degradation by model size](docs/img/degradacion-por-tamano-dark.png#gh-dark-mode-only)

The numbers come from `runs/*.json` and the charts are regenerated with
`python scripts/grafica_comparativa.py` and `python scripts/grafica_gguf.py`.
Details in [`runs/COMPARATIVA.md`](runs/COMPARATIVA.md) and
[`runs/COMPARATIVA_GGUF.md`](runs/COMPARATIVA_GGUF.md).

### What Octuma does **not** do

Quantized models **generate more slowly** in PyTorch: 4.2 tokens/s against
14.4 for the original on Qwen 3B. `QuantLinear` unpacks the 4 bits on every
multiplication without a dedicated kernel, which bitsandbytes and AutoGPTQ do
have. Octuma's case is **memory and quality**, not speed: to run fast, export
to GGUF and use llama.cpp.

It also hasn't been compared against AutoGPTQ or GPTQModel, its most direct
technical rivals: today they don't install with modern setuptools.

## How it works

| Step | What it does |
|---|---|
| **Calibrate** | Takes random windows from a real corpus (wikitext-2, C4 or your own texts) and measures, layer by layer, which input directions matter. It's stored as `H = 2·XXᵀ`. |
| **Quantize** | Splits the weights into blocks of 32 per row, computes a scale and a zero point per block, and brings them to 4 bits. Rounding isn't blind: with GPTQ, each column's error is spread over the remaining columns using the Hessian. |
| **Evaluate** | Perplexity on wikitext-2 with non-overlapping windows, memory per layer type, and tokens per second. |
| **Export** | Its own `.tq` format (packed safetensors) for PyTorch, and **GGUF** for llama.cpp and Android. |

### Data-driven mixed precision

Not every layer suffers the same. `octuma analyze` measures, layer by layer,
how much its **output** changes when quantized, not how much its weights
change:

```
||ΔW·X||² = tr(ΔW · H · ΔWᵀ)
```

With that it ranks layers by real damage and builds a plan: the most sensitive
go up to 8 bits and the rest stay at 4, without going over the average bits
you ask for.

```bash
octuma analyze Qwen/Qwen2.5-0.5B-Instruct --target-bits 4.5 -o plan.json
octuma quantize Qwen/Qwen2.5-0.5B-Instruct --out out/qwen-mix --plan plan.json
```

### Why groups, and why asymmetric

A single scale per tensor gets ruined by a single outlier weight. With groups
of 32 the damage stays inside its block. And since weights are almost never
centered on zero, also storing a zero point (asymmetric) uses all 16 levels
instead of wasting half the range.

The cost is small: each group stores an FP16 scale and a zero point, so ~4.5
bits per weight instead of 4.

## Android and llama.cpp

The exporter writes GGUF with the weights in **Q4_1**, which is exactly the
same format as an Octuma asymmetric group of 32:

```
Octuma:  w = (q - z)·s        Q4_1:  w = d·q + m        d = s,  m = -z·s
```

That's why the exporter requires groups of 32, which is already the default:

```bash
octuma quantize <model>
octuma export <int4-folder> --out model-int4.gguf
python scripts/verify_gguf.py <int4-folder> model-int4.gguf
```

**Always verify before publishing a GGUF.** The weights can be perfect and the
file still come out broken because of the metadata: a wrong `rope_theta`
**doubles the perplexity without touching a single weight**, and the model
keeps answering short sentences normally, so you can't tell at a glance. It
happened to us: the three published models were degraded until
`verify_gguf.py` learned to check the metadata as well as the weights. It
returns an error code, so it can run in CI.

**Actually verified**: Qwen2.5-0.5B quantized with Octuma, exported to GGUF and
run in llama.cpp:

```
$ llama-cli -m qwen05b-int4.gguf -p "La capital de Francia es" --temp 0 -n 32 -st -ngl 0
La capital de Francia es París.

prompt: 229 tok/s · generation: 64 tok/s   (CPU only, 6 threads)
```

It has to be `llama-cli` and not `llama-completion`: these are *Instruct*
models and `llama-cli` applies their chat template. With raw completion, the
0.5B at `--temp 0` just keeps repeating the question.

## Using it from Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from octuma.calibrate import load_wikitext2
from octuma.quantizer import QuantConfig, quantize_model
from octuma.export.tq import save_quantized

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

calib = load_wikitext2(tok, n_samples=64, seq_len=512)
report = quantize_model(model, calib, QuantConfig(bits=4, group_size=32))

print(report.summary())
save_quantized(model, "out/qwen-int4")
```

Loading one that's already quantized:

```python
from transformers import AutoConfig, AutoModelForCausalLM
from octuma.export.tq import load_quantized

cfg = AutoConfig.from_pretrained("out/qwen-int4")
model = load_quantized(AutoModelForCausalLM.from_config(cfg), "out/qwen-int4")
```

Mixed precision, for the layers that suffer most:

```python
cfg = QuantConfig(bits=4, group_size=32, bits_overrides={"mlp.down_proj": 8})
```

## Sweep results

33 runs on Qwen2.5 (0.5B, 1.5B, 3B and 7B), in `runs/*.json`. The ranking of
methods is **the same across all four models**, which is the best sign that the
implementation does what it says:

```
GPTQ+AWQ  >  GPTQ  >  AWQ-RTN  >  RTN
```

Degradation of the best method against FP16: 5.2% (0.5B), 2.3% (1.5B), 2.4%
(3B), 1.8% (7B with GPTQ only: its AWQ runs didn't fit in memory). INT8 is
practically free (+0.03%) but only compresses 1.91x; INT4 compresses 3.66x,
counting only the weights: the whole model shrinks 59% on the 3B, because the
embeddings stay in FP16.

> The sweep ran with **groups of 64**, which was the default back then. Today
> the CLI uses 32, which is what the GGUF exporter requires and what the three
> published models contain. Smaller groups store more scales and so compress
> less: 3.66x with 64 against 3.37x with 32 on the 0.5B. The llama.cpp tables
> above do use 32; what hasn't been measured is how much perplexity changes
> between one group size and the other.

### How much calibration you need

GPTQ estimates that same `H = 2·XXᵀ` per layer. With fewer tokens than the
layer has dimensions, that matrix is singular and error compensation turns into
noise: the result comes out **worse** than not using GPTQ. We measured it on
the 0.5B:

| Calibration tokens | GPTQ | GPTQ + AWQ |
|---|---|---|
| 512 (fewer than the 896 dimensions) | worse than RTN | much worse than RTN |
| 8192 | +7.5% | +4.2% |

That's why `quantize_model` warns you when calibration falls short. Rule of
thumb: at least 10 times the dimension of the widest layer.

### A negative result worth reporting

Scale search (`--search-scale`) helps on 0.5B and 3B, and **hurts** on 1.5B and
7B. On the 7B it reduces the weights' reconstruction error (0.09503 → 0.09152)
but **worsens** perplexity (7.4386 → 7.5189).

Minimizing weight error is not the same as preserving model quality. Details
in [`runs/NOTA_rtn_search.md`](runs/NOTA_rtn_search.md) (in Spanish).

## Status

- [x] Group quantization INT2/INT4/INT8, symmetric and asymmetric
- [x] Real bit packing (INT4 = half a byte)
- [x] GPTQ with error compensation and a damped Hessian
- [x] Sequential block-by-block quantization (error propagates like it does in real use)
- [x] `QuantLinear` layer that dequantizes on the fly
- [x] Per-layer mixed precision (`bits_overrides`)
- [x] Save and load `.tq`
- [x] Perplexity, memory and speed
- [x] Export to GGUF Q4_1 / Q8_0 (llama.cpp / Android)
- [x] Per-layer sensitivity analysis and mixed-precision plan
- [x] GGUF validated running in llama.cpp and measured against its formats
- [ ] KV cache quantization
- [ ] Fast INT4 kernel (today it dequantizes on the fly)
- [ ] Android demo app

## Development

```bash
git clone https://github.com/1mano1/octuma.git
cd octuma
pip install -e ".[hf,gguf,dev]"
pytest -q          # 74 tests, seconds on a CPU
ruff check .
```

The tests use tiny Llama models built on the fly, without downloading
anything. The two rules that already cost us are in
[`CONTRIBUTING.md`](CONTRIBUTING.md) (in Spanish): a default never hides a
failure, and the defaults are whatever measured best.

## License

Octuma's code is **MIT** (see [LICENSE](LICENSE)).

**The models are another matter.** A quantized model is a derivative work: it
keeps the original's license, and Octuma's doesn't loosen it. That's why each
published repo carries its base model's license:

| Base | License | Commercial use |
|---|---|---|
| Qwen2.5-0.5B / 1.5B / 7B-Instruct | Apache 2.0 | yes |
| Qwen2.5-3B-Instruct | [Qwen Research](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) | **no** |

If you quantize another model with Octuma, check its license before publishing
it: several popular families (Llama, Gemma) come with their own terms that
travel with the derived weights.

Built with Qwen.
