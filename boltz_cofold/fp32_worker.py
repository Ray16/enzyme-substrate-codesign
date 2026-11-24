"""Subprocess entry point for FP32 serial-capped Boltz-2 predictions."""
from __future__ import annotations

import json

from .predict import patch_boltz_fp32_serial


def main() -> None:
    info = patch_boltz_fp32_serial()
    print(json.dumps(info), flush=True)
    import boltz.main as engine

    engine.cli()


if __name__ == "__main__":
    main()
