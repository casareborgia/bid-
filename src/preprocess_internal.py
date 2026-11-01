"""사내 공사용역계약대장 CSV 정제 (Phase 1).

원본: data/raw/공사용역계약대장(13년치).csv (CP949, 헤더 없음, 90컬럼)
출력: data/internal_clean_post2019.parquet / .csv, out/excluded_sample.csv

규칙은 CLAUDE.md "데이터" 섹션을 따른다.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "raw" / "공사용역계약대장(13년치).csv"
DEFAULT_OUT_DIR = ROOT / "data"

EXPECTED_COLS = 90

# (컬럼 인덱스, 출력명)
COLUMN_MAP = {
    0: "dept_name",
    1: "contract_no",
    2: "contract_date",
    3: "work_type_txt",
    4: "contract_method",
    5: "award_method",
    11: "contract_amt",
    12: "bid_rate_yega",
    14: "design_amt",
    16: "est_price",
    18: "base_amt",
    19: "yega_amt",
    39: "duration_days",
    52: "bidder_count",
}
NUMERIC_COLS = ["contract_amt", "bid_rate_yega", "design_amt", "est_price", "base_amt", "yega_amt"]

MIN_YEAR = 2019
RATIO_TOL = 0.001
RATE_MIN, RATE_MAX = 96.5, 103.5
WORK_TYPE_CODE = {"공사": 1, "용역": 2}


def parse_number(s: str) -> float:
    """'14,661,820,800' -> 14661820800.0, 빈 문자열/비숫자 -> NaN."""
    s = (s or "").replace(",", "").strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_duration(s: str) -> float:
    """'730일' -> 730, 숫자 없으면 NaN, 0~3650 클리핑."""
    m = re.search(r"\d+", s or "")
    if not m:
        return np.nan
    return float(min(max(int(m.group()), 0), 3650))


def load_raw(path: Path) -> pd.DataFrame:
    """CP949 바이트를 읽어 90컬럼 CSV 행만 DataFrame으로 만든다."""
    text = path.read_bytes().decode("cp949", errors="ignore")
    rows, bad = [], 0
    for line in text.splitlines():
        if '","' not in line:
            continue
        try:
            r = next(csv.reader([line]))
        except csv.Error:
            bad += 1
            continue
        if len(r) != EXPECTED_COLS:
            bad += 1
            continue
        rows.append(r)
    if bad:
        print(f"[경고] 컬럼 수가 {EXPECTED_COLS}개가 아닌 행 {bad}건 제외", file=sys.stderr)
    raw = pd.DataFrame(rows)
    df = pd.DataFrame({name: raw[idx] for idx, name in COLUMN_MAP.items()})
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """타입 변환 → 파생 컬럼 → 학습 필터. (정제본, 제외샘플(비율 불일치), 통계) 반환."""
    df = df.copy()
    for c in NUMERIC_COLS:
        df[c] = df[c].map(parse_number).astype(float)
    df["duration_days"] = df["duration_days"].map(parse_duration).astype(float)
    df["bidder_count"] = df["bidder_count"].map(parse_number).fillna(1).astype(float)
    df["contract_date"] = pd.to_datetime(df["contract_date"].str.strip(), format="%Y/%m/%d", errors="coerce")
    df["year"] = df["contract_date"].dt.year
    df["month"] = df["contract_date"].dt.month
    df["work_type"] = df["work_type_txt"].str.strip().map(WORK_TYPE_CODE)
    df["work_type_from_no"] = pd.to_numeric(df["contract_no"].str.strip().str[6], errors="coerce")
    df["branch_code"] = df["contract_no"].str.strip().str[:4]

    df["base_design_ratio"] = df["base_amt"] / df["design_amt"]
    df["yega_rate"] = df["yega_amt"] / df["base_amt"] * 100.0
    df["is_year_end"] = df["month"].isin([11, 12]).astype(int)

    stats: dict = {"raw_rows": int(len(df))}

    post = df[df["year"] >= MIN_YEAR]
    stats["rows_post2019"] = int(len(post))
    has_base = post[post["base_amt"] > 0]
    stats["rows_base_zero_or_blank"] = int(len(post) - len(has_base))

    ratio_ok = (has_base["base_design_ratio"] - 1.0).abs() < RATIO_TOL
    excluded_ratio = has_base[~ratio_ok]
    stats["rows_ratio_mismatch"] = int(len(excluded_ratio))

    cand = has_base[ratio_ok]
    in_range = cand["yega_rate"].between(RATE_MIN, RATE_MAX)
    stats["rows_rate_out_of_range"] = int((~in_range).sum())
    out = cand[in_range].sort_values(["contract_date", "contract_no"]).reset_index(drop=True)
    stats["rows_clean"] = int(len(out))

    # 기초금액이 있는 수의계약은 필터를 통과한다. 학습에서는 train_model.split_negotiated 가 분리한다.
    stats["rows_negotiated_kept"] = int(out["contract_method"].fillna("").str.strip().str.startswith("수의").sum())

    mismatch = (out["work_type"] != out["work_type_from_no"]).sum()
    stats["work_type_mismatch_vs_contract_no"] = int(mismatch)
    stats["branch_codes"] = out["branch_code"].value_counts().to_dict()

    r = out["yega_rate"]
    stats["yega_rate"] = {
        "mean": round(float(r.mean()), 4),
        "std": round(float(r.std(ddof=0)), 4),
        "min": round(float(r.min()), 3),
        "q25": round(float(r.quantile(0.25)), 3),
        "q50": round(float(r.quantile(0.50)), 3),
        "q75": round(float(r.quantile(0.75)), 3),
        "max": round(float(r.max()), 3),
    }
    stats["by_year"] = out.groupby("year").size().to_dict()
    stats["by_work_type"] = out.groupby("work_type_txt")["yega_rate"].agg(["size", "mean"]).round(4).to_dict("index")
    stats["by_award_method"] = out.groupby("award_method")["yega_rate"].agg(["size", "mean"]).round(4).to_dict("index")
    stats["bidder_count_median"] = float(out["bidder_count"].median())
    stats["est_price_median"] = float(out["est_price"].median())

    keep = [
        "contract_no", "contract_date", "year", "month", "dept_name", "work_type", "work_type_txt",
        "contract_method", "award_method", "est_price", "design_amt", "base_amt", "yega_amt",
        "yega_rate", "contract_amt", "bid_rate_yega", "duration_days", "bidder_count", "is_year_end",
    ]
    out = out[keep].copy()
    out["source"] = "internal"
    return out, excluded_ratio, stats


def print_report(stats: dict) -> None:
    print("=== 사내 계약대장 정제 결과 ===")
    print(f"원본 행수            : {stats['raw_rows']:,}")
    print(f"{MIN_YEAR}년 이후        : {stats['rows_post2019']:,}")
    print(f"  - 기초금액 0/공란 제외: {stats['rows_base_zero_or_blank']:,} (수의계약 등)")
    print(f"  - 기초≠설계 제외     : {stats['rows_ratio_mismatch']:,}")
    print(f"  - 사정률 범위 밖 제외 : {stats['rows_rate_out_of_range']:,}")
    print(f"최종 정제본           : {stats['rows_clean']:,} (이 중 수의계약 {stats['rows_negotiated_kept']:,}건은 학습 시 분리됨)")
    print(f"work_type 불일치(계약번호 7번째 자리 대비): {stats['work_type_mismatch_vs_contract_no']}")
    print(f"branch_code 분포      : {stats['branch_codes']}")
    y = stats["yega_rate"]
    print(f"사정률(%)  mean {y['mean']} | std {y['std']} | min {y['min']} | Q25 {y['q25']} | Q50 {y['q50']} | Q75 {y['q75']} | max {y['max']}")
    print("연도별 건수           :", {int(k): int(v) for k, v in stats["by_year"].items()})
    print("공사/용역별 (건수, 평균):", stats["by_work_type"])
    print("낙찰자결정방법별       :", stats["by_award_method"])
    print(f"참여업체수 중앙값 {stats['bidder_count_median']} | 추정가격 중앙값 {stats['est_price_median']/1e8:.2f}억")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사내 계약대장 CSV 정제")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"입력 파일 없음: {args.input}", file=sys.stderr)
        return 1

    df = load_raw(args.input)
    out, excluded, stats = clean(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    out.to_parquet(args.out_dir / "internal_clean_post2019.parquet", index=False)
    out.to_csv(args.out_dir / "internal_clean_post2019.csv", index=False, encoding="utf-8-sig")
    excluded_cols = ["contract_no", "contract_date", "dept_name", "work_type_txt", "contract_method",
                     "award_method", "design_amt", "base_amt", "yega_amt", "base_design_ratio"]
    excluded[excluded_cols].head(10).to_csv(ROOT / "out" / "excluded_sample.csv", index=False, encoding="utf-8-sig")

    print_report(stats)
    print(f"\n저장: {args.out_dir / 'internal_clean_post2019.parquet'} ({len(out):,}건)")
    print(f"제외 샘플: out/excluded_sample.csv (기초≠설계 {len(excluded)}건 중 10건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
