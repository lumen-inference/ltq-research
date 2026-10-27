"""
Training tables for production TCQ formats with tail-biting and extended labels (night 2) -> results/tcq_x_tables.json.

Key: format name; value: K, h, gain step, extra-parity masks, table in the kernel layout [nl, 2^(K-1)]
and the Gaussian gap to R-D (fair bench `src/universal_bench.py`). For K = 4 (4 subsets) the tables are taken
from results/universal_gaussian.json (already trained for tail-biting).

    python -m src.train_tcq_x
"""

import json
import math
from pathlib import Path

from src.universal_bench import gap_tcq, train_tcq_x

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tcq_x_tables.json"

# name: (K, h, steps, masks)
FORMATS = {
    "TCQ2X0": (2, 0, (0.0,), (58, 299)),
    "TCQ2X": (2, 1, (0.4,), (58, 299)),
    "TCQ3X1": (3, 1, (0.4,), (241,)),
    "TCQ3X": (3, 2, (0.22, 0.3), (241,)),
    "TCQ3X3": (3, 3, (0.1, 0.15), (241,)),
    "TCQ3X16": (3, 2, (0.22, 0.3), (211, 397)),
    # 64-entry tables (night 2, mask search): K = 3 / 16 subsets, K = 4 / 8 subsets
    "TCQ3Y0": (3, 0, (0.0,), (40, 475)),
    "TCQ3Y1": (3, 1, (0.4,), (40, 475)),
    "TCQ3Y2": (3, 2, (0.22, 0.3), (40, 475)),
    "TCQ4Y0": (4, 0, (0.0,), (462,)),
    "TCQ4Y1": (4, 1, (0.4,), (462,)),
    "TCQ4Y2": (4, 2, (0.22, 0.3), (146,)),
    "TCQ4Y3": (4, 3, (0.1, 0.15), (146,)),
}
# header per 32 weights (two tile rows): name -> (K, h, steps, masks, sub)
FORMATS_W = {
    "TCQ3W2": (3, 2, (0.22, 0.3), (241,), 32),
    "TCQ3W3": (3, 3, (0.1, 0.15), (241,), 32),
    "TCQ4W2": (4, 2, (0.22, 0.3), (), 32),
    "TCQ4W3": (4, 3, (0.1, 0.15), (), 32),
    "TCQ4W4": (4, 4, (0.06, 0.09), (), 32),
}
# half-integer (pair parity, D2 lattice): format table K, rate K - 0.5: name -> (K, h, steps, masks, sub, pp)
FORMATS_PP = {
    "TCQ25X": (3, 1, (0.4,), (241,), 16, True),
    "TCQ35X1": (4, 1, (0.4,), (), 16, True),
    "TCQ35X": (4, 2, (0.22, 0.3), (), 16, True),
    "TCQ45X": (5, 2, (0.22, 0.3), (), 16, True),
    "TCQ45X3": (5, 3, (0.1, 0.15), (), 16, True),
    # 64-entry half-integer
    "TCQ25Y": (3, 1, (0.4,), (40, 475), 16, True),
    "TCQ35Y1": (4, 1, (0.4,), (462,), 16, True),
    "TCQ35Y": (4, 2, (0.22, 0.3), (146,), 16, True),
    # quarter-rate (even sum of the low bits of a group of 4, D4 lattice): table K, rate K - 0.25
    "TCQ275X": (3, 1, (0.4,), (241,), 16, 2),
    "TCQ275X2": (3, 2, (0.22, 0.3), (241,), 16, 2),
    "TCQ375X": (4, 2, (0.22, 0.3), (), 16, 2),
    "TCQ475X": (5, 2, (0.22, 0.3), (), 16, 2),
    # K + 0.25 (all low bits of a group of 4 are equal): table K, rate K - 0.75
    "TCQ225X": (3, 1, (0.4,), (241,), 16, 3),
    "TCQ325X": (4, 2, (0.22, 0.3), (), 16, 3),
    "TCQ425X": (5, 2, (0.22, 0.3), (), 16, 3),
}


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = {n: v + (16, False) for n, v in FORMATS.items()} | {n: v + (False,) for n, v in FORMATS_W.items()} | FORMATS_PP
    for name, (K, h, steps, mx, sub, pp) in todo.items():
        if name in res:
            continue
        best = None
        for st in steps:
            for spread in ((1.0, math.sqrt(3.0)) if pp else (1.0,)):
                tab, gains, nc = train_tcq_x(K, h, st, mx, sub=sub, pp=pp, spread=spread)
                gp = gap_tcq(K, tab, gains, h, nc, m_extra=mx, sub=sub, pp=pp)
                if best is None or gp < best["gap"]:
                    best = {"K": K, "h": h, "step": st, "m_extra": list(mx), "gap": gp, "sub": sub, "pp": pp,
                            "lut": [round(v, 5) for v in tab.tolist()]}
        res[name] = best
        print(f"{name}: K={K - {0: 0.0, 1: 0.5, 2: 0.25, 3: 0.75}[int(pp)]} h={h} per {sub} weights, step {best['step']}, masks {mx}: gap {best['gap']:.3f}",
              flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    uni = json.loads((ROOT / "results" / "universal_gaussian.json").read_text())
    # K = 4 and K = 5 (4 subsets): tables already trained for tail-biting in universal_gaussian.json; K = 5 has 64 levels
    # (on CPU: PWL 2 x 32, exact for K = 5: 2 levels per segment)
    for K, h, name in ((4, 1, "TCQ4X1"), (4, 2, "TCQ4X2"), (4, 3, "TCQ4X"), (5, 3, "TCQ5X3"), (5, 4, "TCQ5X")):
        u = uni[f"tcq_K{K}_h{h}"]
        res[name] = {"K": K, "h": h, "step": u["step"], "m_extra": [], "gap": u["gap"], "lut": [round(v, 5) for v in u["tab"]]}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
