"""Control de pods de RunPod desde la terminal.

    python scripts/runpod.py gpus                 # que hay disponible y a cuanto
    python scripts/runpod.py list                 # pods activos
    python scripts/runpod.py create --gpu "NVIDIA A40" --yes
    python scripts/runpod.py stop <pod_id>
    python scripts/runpod.py terminate <pod_id>

La clave se lee de RUNPOD_API_KEY (o del archivo .env) y nunca se imprime.
Crear un pod cuesta dinero: `create` exige --yes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

API = "https://api.runpod.io/graphql"


def load_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in ("RUNPOD_API_KEY", "RUNPOD_TOKEN"):
                    return v.strip().strip('"').strip("'")
            elif line.startswith("rpa_"):
                return line  # clave suelta en el .env
    sys.exit("No encontre RUNPOD_API_KEY (ponla en .env o en el entorno)")


def gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(
        API,
        params={"api_key": load_key()},
        json={"query": query, "variables": variables or {}},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        sys.exit(json.dumps(data["errors"], indent=2))
    return data["data"]


def cmd_gpus(args) -> None:
    data = gql(
        """
        query { gpuTypes { id displayName memoryInGb
          lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice } } }
        """
    )
    rows = [g for g in data["gpuTypes"] if g["memoryInGb"] >= args.min_vram]
    rows.sort(key=lambda g: (g["lowestPrice"]["uninterruptablePrice"] or 99))
    print(f"{'GPU':32} {'VRAM':>6} {'on-demand':>11} {'spot':>8}")
    for g in rows[: args.limit]:
        p = g["lowestPrice"]
        od = p["uninterruptablePrice"]
        sp = p["minimumBidPrice"]
        print(
            f"{g['displayName'][:32]:32} {g['memoryInGb']:5}G "
            f"{f'${od:.3f}/h' if od else '-':>11} {f'${sp:.3f}/h' if sp else '-':>8}"
        )


def cmd_list(args) -> None:
    data = gql(
        """
        query { myself { pods { id name desiredStatus costPerHr
          machine { gpuDisplayName } runtime { uptimeInSeconds } } } }
        """
    )
    pods = data["myself"]["pods"]
    if not pods:
        print("sin pods")
        return
    for p in pods:
        up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        print(
            f"{p['id']}  {p['name'][:24]:24} {p['desiredStatus']:10} "
            f"{p['machine']['gpuDisplayName']:20} ${p['costPerHr']}/h  {up / 3600:.1f}h"
        )


def _env_list(args) -> list[dict]:
    """Variables para el contenedor: llave SSH y credenciales opcionales."""
    env: list[dict] = []
    if args.pubkey:
        key = Path(args.pubkey).expanduser().read_text(encoding="utf-8").strip()
        env.append({"key": "PUBLIC_KEY", "value": key})
    hf = os.environ.get("HF_TOKEN") or _from_env_file("HF_TOKEN")
    if hf:
        env.append({"key": "HF_TOKEN", "value": hf})
    return env


def _from_env_file(name: str) -> str | None:
    env = Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def cmd_create(args) -> None:
    if not args.yes:
        sys.exit("Crear un pod cuesta dinero: repite el comando con --yes")
    data = gql(
        """
        mutation($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) { id name costPerHr machine { gpuDisplayName } }
        }
        """,
        {
            "input": {
                "cloudType": "SECURE",
                "gpuCount": 1,
                "gpuTypeId": args.gpu,
                "name": args.name,
                "imageName": args.image,
                "containerDiskInGb": args.disk,
                "volumeInGb": args.volume,
                "volumeMountPath": "/workspace",
                "ports": "8888/http,22/tcp",
                "minVcpuCount": 8,
                "minMemoryInGb": 32,
                "dockerArgs": "",
                "env": _env_list(args),
            }
        },
    )
    pod = data["podFindAndDeployOnDemand"]
    print(f"pod {pod['id']} · {pod['machine']['gpuDisplayName']} · ${pod['costPerHr']}/h")
    print("Entra por SSH o por la web y corre:")
    print("  bash scripts/runpod_bootstrap.sh")


def cmd_stop(args) -> None:
    gql("mutation($id: String!) { podStop(input: {podId: $id}) { id } }", {"id": args.pod_id})
    print(f"pod {args.pod_id} detenido (el volumen se sigue cobrando)")


def cmd_terminate(args) -> None:
    gql(
        "mutation($id: String!) { podTerminate(input: {podId: $id}) }",
        {"id": args.pod_id},
    )
    print(f"pod {args.pod_id} eliminado")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gpus", help="lista GPUs y precios")
    g.add_argument("--min-vram", type=int, default=24)
    g.add_argument("--limit", type=int, default=15)
    g.set_defaults(func=cmd_gpus)

    listar = sub.add_parser("list", help="pods activos")
    listar.set_defaults(func=cmd_list)

    c = sub.add_parser("create", help="crea un pod")
    c.add_argument("--gpu", default="NVIDIA A40")
    c.add_argument("--name", default="tinyq-sweep")
    c.add_argument(
        "--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    )
    c.add_argument("--disk", type=int, default=40)
    c.add_argument("--volume", type=int, default=100)
    c.add_argument(
        "--pubkey",
        default="~/.ssh/id_ed25519.pub",
        help="llave publica que se instala en el pod para entrar por SSH",
    )
    c.add_argument("--yes", action="store_true")
    c.set_defaults(func=cmd_create)

    s = sub.add_parser("stop", help="detiene un pod")
    s.add_argument("pod_id")
    s.set_defaults(func=cmd_stop)

    t = sub.add_parser("terminate", help="elimina un pod")
    t.add_argument("pod_id")
    t.set_defaults(func=cmd_terminate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
