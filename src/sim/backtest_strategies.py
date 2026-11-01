"""투찰 전략 재연 백테스트 (시뮬레이터 2단계 검증).

실제 입찰마다 실제 사정률 y 와 실제 경쟁사 투찰(환산 사정률)을 그대로 두고, 전략이 정한 내 환산 사정률을 끼워 넣어
1순위가 됐을지 센다(sim.optimize.replay_win). 시간순 앞 절반으로 분포를 적합하고 뒤 절반에서 평가한다.

전략
  Q25/Q50/Q75 : 학습 구간 실제 사정률의 해당 분위수(예가 범위별) — 지금 예측 도구의 기준별 투찰가와 같은 논리
  MC 최적     : optimize.optimal_rate(y = 학습 구간 사정률, F = 학습 구간 경쟁사 환산 사정률, N = 그 입찰의 경쟁사 수)
  MC 재표본   : 경쟁사를 독립 분포 대신 학습 구간 입찰(같은 예가 범위·같은 경쟁사 수 구간)의 투찰 묶음으로 재표본
                (optimize.win_prob_curve_bootstrap — 입찰 안 상관과 경쟁사 수를 그대로 반영)
  무작위 입찰자: 경쟁사 중 한 명이 될 확률의 기준선 1/(N+1)

사용 예
  uv run python src/sim/backtest_strategies.py --month 2026-10 --work servc
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import collect_g2b_api as cg  # noqa: E402
from sim import collect_bidders as CB  # noqa: E402
from sim import optimize as O  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
N_BUCKETS = [0, 10, 30, 100, 10**6]


def y_history_before(month: str, parquet: Path = cg.OUT_DIR / "g2b_detail_all.parquet") -> dict[str, np.ndarray]:
    """평가월 이전 시장 데이터의 예가 범위별 실제 사정률(미래 참조 없음)."""
    g = pd.read_parquet(parquet, columns=["ym", "range_cls", "yega_rate"])
    g = g[(g["ym"] < month) & g["yega_rate"].between(96.5, 103.5)]
    return {rc: g.loc[g["range_cls"] == rc, "yega_rate"].to_numpy() for rc in ("±2", "±3")}


def load_auctions(month: str, work: str, raw_dir: Path = cg.RAW_DIR) -> pd.DataFrame:
    """입찰 1건 1행: 기초금액·실제 사정률·하한율·예가 범위·개찰일·경쟁사 환산 사정률 배열. 용역(A값 없음) 기준."""
    t = CB.targets(month, work)
    rows = [json.loads(l) for l in (raw_dir / f"compt_{month}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    c = pd.DataFrame(rows)
    c["bid_key"] = c[cg.KEY_COLS].astype(str).apply(lambda s: s.str.strip()).agg("|".join, axis=1)
    c["amt"] = pd.to_numeric(c["bidprcAmt"], errors="coerce")
    c = c[c["amt"] > 0]
    m = c.merge(t[["bid_key", "base_amt", "yega_rate", "notice_lower", "range_cls", "open_dt"]], on="bid_key")
    m["r"] = O.implied_rate(m["amt"], m["base_amt"], m["notice_lower"])
    m["below"] = m["rmrk"].astype(str).str.contains("미달")
    a = m.groupby("bid_key").agg(base_amt=("base_amt", "first"), y=("yega_rate", "first"), ll=("notice_lower", "first"),
                                 range_cls=("range_cls", "first"), open_dt=("open_dt", "first"),
                                 n=("r", "size")).reset_index()
    a["rates"] = a["bid_key"].map(m.groupby("bid_key")["r"].apply(lambda s: s.to_numpy()))
    # 일관성: 하한 미달 표기 ⇔ r < y (원 단위 올림 오차 허용)
    chk = m.assign(pred_below=m["r"] < m["yega_rate"] - 1e-6)
    a.attrs["below_consistency"] = float((chk["pred_below"] == chk["below"]).mean())
    return a.sort_values("open_dt").reset_index(drop=True)


def evaluate(auctions: pd.DataFrame, y_samples_n: int = 4000, seed: int = 0,
             y_history: dict[str, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict]:
    """y_history: 예가 범위별 과거 실제 사정률(평가월 이전 시장 데이터). 없으면 학습 절반의 사정률을 쓴다."""
    half = len(auctions) // 2
    train, test = auctions.iloc[:half], auctions.iloc[half:].copy()
    rng = np.random.default_rng(seed)
    y_train = y_history or {rc: g["y"].to_numpy() for rc, g in train.groupby("range_cls")}
    comp_train = {rc: np.concatenate(g["rates"].to_list()) for rc, g in train.groupby("range_cls")}
    y_samp = {rc: rng.choice(v, size=y_samples_n, replace=True) for rc, v in y_train.items()}
    quant = {rc: {q: float(np.percentile(v, q)) for q in (25, 50, 75)} for rc, v in y_train.items()}

    @lru_cache(maxsize=None)
    def mc_rate(rc: str, n: int) -> float:
        return O.optimal_rate(y_samp[rc], comp_train[rc], n, lo=98.0, hi=102.0, step=0.01)["rate"]

    grid = np.round(np.arange(98.0, 102.0 + 1e-9, 0.01), 4)
    train_b = train.assign(nb=pd.cut(train["n"], N_BUCKETS, right=False, labels=False))

    @lru_cache(maxsize=None)
    def boot_rate(rc: str, nb: int) -> float:
        pool = train_b[(train_b["range_cls"] == rc) & (train_b["nb"] == nb)]["rates"].to_list()
        if len(pool) < 5:                                                # 표본이 적으면 같은 예가 범위 전체
            pool = train_b[train_b["range_cls"] == rc]["rates"].to_list()
        p = O.win_prob_curve_bootstrap(grid, y_samp[rc], pool)
        return float(grid[int(np.argmax(p))])

    recs = []
    for row in test.itertuples():
        if row.range_cls not in y_train:
            continue
        rates = {f"Q{q}": quant[row.range_cls][q] for q in (25, 50, 75)}
        rates["MC 최적"] = mc_rate(row.range_cls, int(row.n))
        nb = int(pd.cut([row.n], N_BUCKETS, right=False, labels=False)[0])
        rates["MC 재표본"] = boot_rate(row.range_cls, nb)
        rec = {"bid_key": row.bid_key, "range_cls": row.range_cls, "n": int(row.n), "random": 1.0 / (row.n + 1)}
        for k, r in rates.items():
            rec[k] = O.replay_win(r, row.y, row.rates)
            rec[f"{k}_valid"] = r >= row.y
        recs.append(rec)
    res = pd.DataFrame(recs)
    res["n_bucket"] = pd.cut(res["n"], N_BUCKETS, right=False,
                             labels=["<10", "10~29", "30~99", "100+"])
    meta = {"n_train": len(train), "n_test": len(res), "quantiles": quant,
            "y_source": "과거 시장 데이터" if y_history else "학습 절반"}
    return res, meta


def write_report(res: pd.DataFrame, meta: dict, consistency: float, path: Path) -> str:
    strat = ["Q25", "Q50", "Q75", "MC 최적", "MC 재표본"]
    def table(df: pd.DataFrame) -> str:
        row = {"n": len(df), "무작위 기준 1/(N+1)": f"{df['random'].mean()*100:.2f}%"}
        for s in strat:
            p = df[s].mean()
            ci = 1.96 * np.sqrt(p * (1 - p) / max(len(df), 1))              # 95% 신뢰구간 반폭(정규 근사)
            row[s] = f"{p*100:.1f}±{ci*100:.1f}% (유효 {df[s + '_valid'].mean()*100:.0f}%)"
        return row
    lines = ["# 투찰 전략 재연 백테스트", "",
             f"- 학습(앞 절반) {meta['n_train']}건 → 평가(뒤 절반) {meta['n_test']}건. 실제 사정률 분포: {meta['y_source']}. 실제 사정률·실제 경쟁사 투찰을 그대로 두고 내 투찰을 끼워 넣어 1순위 여부를 센다.",
             f"- 데이터 점검: '낙찰하한선 미달' 표기와 환산 사정률 판정 일치율 {consistency*100:.2f}%",
             "- 셀 값: 낙찰률(1순위 비율) ± 95% 신뢰구간 반폭, 괄호는 하한 통과율. 신뢰구간이 겹치면 차이를 단정할 수 없다.", ""]
    rows = [{"구분": "전체", **table(res)}]
    rows += [{"구분": f"예가 {k}", **table(g)} for k, g in res.groupby("range_cls")]
    rows += [{"구분": f"경쟁사 {k}", **table(g)} for k, g in res.groupby("n_bucket", observed=True)]
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines += ["", "학습 구간 사정률 분위수(예가 범위별): " + json.dumps(meta["quantiles"], ensure_ascii=False)]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="투찰 전략 재연 백테스트")
    ap.add_argument("--month", required=True)
    ap.add_argument("--work", choices=["servc"], default="servc")
    args = ap.parse_args(argv)
    auctions = load_auctions(args.month, args.work)
    res, meta = evaluate(auctions, y_history=y_history_before(args.month))
    path = OUT / f"sim_strategy_backtest_{args.month}.md"
    print(write_report(res, meta, auctions.attrs["below_consistency"], path))
    print(f"리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
