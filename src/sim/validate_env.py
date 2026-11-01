"""입찰 환경 시뮬레이터 검증: 시뮬레이션 사정률 분포 vs 실측(나라장터 수집 parquet) → out/sim_env_check.md

사용 예
  uv run python src/sim/validate_env.py              # 기본 n=400,000, 입찰자 모사 20,000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim import env as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
G2B_PATH = ROOT / "data" / "g2b_detail_all.parquet"
OUT = ROOT / "out" / "sim_env_check.md"
PCTS = (5, 25, 50, 75, 95)


def describe(x: np.ndarray) -> dict:
    q = np.percentile(x, PCTS)
    return {"n": len(x), "mean": x.mean(), "std": x.std(), **{f"Q{p}": v for p, v in zip(PCTS, q)}}


def compare(actual_by_range: dict[int, np.ndarray], n: int, n_agent: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rc, a in actual_by_range.items():
        rows.append({"range": f"±{rc}", "kind": "실측", **describe(a)})
        rows.append({"range": f"±{rc}", "kind": "시뮬(보정 β)", **describe(E.simulate_yega_rates(rc, n, rng))})
        rows.append({"range": f"±{rc}", "kind": "시뮬(β=0)", **describe(E.simulate_yega_rates(rc, n, rng, draw_bias=0.0))})
        rows.append({"range": f"±{rc}", "kind": "시뮬(입찰자 30명 번호 선택)",
                     **describe(E.simulate_yega_rates(rc, n_agent, rng, n_bidders=30))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="입찰 환경 시뮬레이터 vs 실측 사정률 분포")
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--n-agent", type=int, default=20_000)
    args = ap.parse_args(argv)
    emp = E.EmpiricalRates.from_parquet(G2B_PATH)
    df = compare(emp.rates, args.n, args.n_agent)
    fmt = df.copy()
    for c in fmt.columns[3:]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")
    fmt["n"] = df["n"].map(lambda v: f"{v:,}")
    text = ("# 입찰 환경 시뮬레이터 검증\n\n"
            "실측: `data/g2b_detail_all.parquet` 정상 범위(사정률 96.5~103.5). 규칙은 `src/sim/env.py` 문서 참고.\n\n"
            + fmt.to_markdown(index=False) + "\n")
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"리포트: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
