"""Calibracion: muestras reales que alimentan la estimacion de la Hessiana."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

# datasets >= 5 exige el id completo del repositorio
WIKITEXT_REPO = "Salesforce/wikitext"


@dataclass
class CalibrationSet:
    """Lotes de ids listos para pasar por el modelo."""

    batches: list[torch.Tensor]
    seq_len: int
    source: str

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def n_tokens(self) -> int:
        return sum(b.numel() for b in self.batches)


def build_from_texts(
    texts: Iterable[str],
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 1,
    seed: int = 0,
    source: str = "custom",
) -> CalibrationSet:
    """Concatena textos y recorta ventanas aleatorias de `seq_len` tokens.

    Tomar ventanas al azar de un corpus largo evita sesgar la calibracion hacia
    los primeros documentos, que es lo que pasa si solo truncas los primeros N.
    """
    joined = "\n\n".join(t for t in texts if t and t.strip())
    ids = tokenizer(joined, return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len + 1:
        raise ValueError(
            f"el corpus tiene {ids.numel()} tokens, se necesitan al menos {seq_len + 1}"
        )

    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq_len - 1, (n_samples,), generator=gen)
    windows = [ids[s : s + seq_len].unsqueeze(0) for s in starts.tolist()]

    batches = [
        torch.cat(windows[i : i + batch_size], dim=0)
        for i in range(0, len(windows), batch_size)
    ]
    return CalibrationSet(batches=batches, seq_len=seq_len, source=source)


def load_wikitext2(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 1,
    seed: int = 0,
    split: str = "train",
) -> CalibrationSet:
    """Descarga wikitext-2 y arma el conjunto de calibracion."""
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split=split)
    return build_from_texts(
        ds["text"],
        tokenizer,
        n_samples=n_samples,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
        source=f"wikitext-2/{split}",
    )


def load_calibration(
    name: str,
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 1,
    seed: int = 0,
) -> CalibrationSet:
    """Resuelve un nombre de dataset o una ruta a archivos de texto."""
    if name in {"wikitext2", "wikitext-2"}:
        return load_wikitext2(tokenizer, n_samples, seq_len, batch_size, seed)
    if name == "c4":
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        return build_from_texts(
            ds["text"][:20000],
            tokenizer,
            n_samples=n_samples,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            source="c4",
        )

    from pathlib import Path

    path = Path(name)
    if path.exists():
        files: Sequence[Path] = [path] if path.is_file() else sorted(path.glob("**/*.txt"))
        texts = [f.read_text(encoding="utf-8", errors="ignore") for f in files]
        return build_from_texts(
            texts,
            tokenizer,
            n_samples=n_samples,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            source=str(path),
        )
    raise ValueError(f"no se reconoce el dataset de calibracion: {name}")
