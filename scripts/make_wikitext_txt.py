"""Escribe el wikitext-2 test como texto plano para `llama-perplexity`.

El evaluador de Python (`octuma.evaluate.wikitext2_ids`) une el split con
"\\n\\n". Este script hace exactamente lo mismo en un .txt, que es la unica
forma de que el numero de llama.cpp y el de Octuma midan sobre el mismo texto.

    python scripts/make_wikitext_txt.py --out out/wikitext2.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/wikitext2.txt"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset

    from octuma.calibrate import WIKITEXT_REPO

    ds = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split=args.split)
    texto = "\n\n".join(ds["text"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(texto, encoding="utf-8")
    print(f"{args.out} -> {len(texto):,} caracteres, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
