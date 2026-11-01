"""피처 생성 (학습·추론 공용). 규칙은 CLAUDE.md "모델" 섹션.

- 날짜 오름차순 정렬 후 이동통계는 반드시 .shift(1) → 미래 참조 금지.
- 범주 인코더는 학습 구간에서만 적합(fit)하고, 미지 범주는 "기타"로 보낸다.
- `source` 컬럼이 있으면 이동통계는 source(사내/g2b)별로 따로 계산한다. 시장 데이터의 직전 사정률이
  사내 행의 ma_15/std_15 로 섞이지 않게 하기 위해서다. source 컬럼이 없으면 전체를 한 계열로 본다.
- branch_code(상수), base_amt/design_amt(상수 1.0), bid_rate_yega(결과 변수)는 쓰지 않는다.
- 시장(나라장터) 모델은 MARKET_FEATURES 를 쓴다. 사정률 분포를 결정하는 것은 예가 범위(range_cls, ±2%/±3%)이고
  공사/용역·참여업체수·규모는 0.1%p 이하의 영향이라 이동통계·부서·낙찰방법은 넣지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "yega_rate"
DATE_COL = "contract_date"       # 공통 날짜 컬럼. g2b 행은 train_model 에서 open_dt 를 이 이름으로 맞춘다.
SOURCE_COL = "source"            # 있으면 이동통계를 이 값별로 따로 계산 ("internal" / "g2b")
ROLL_WINDOW = 15
ROLL_MIN_PERIODS = 5
RARE_MIN_COUNT = 30
OTHER = "기타"

CAT_FEATURES = ["work_type", "dept_code", "award_method"]
NUM_FEATURES = ["est_price_log", "bidder_count", "is_year_end", "ma_15_rate", "std_15_rate"]
FEATURES = CAT_FEATURES + NUM_FEATURES

RANGE_COL = "range_cls"                      # 예가 범위 분류 (collect_g2b_api.build 가 15개 예비가격에서 계산)
RANGE_CLASSES = ["±2", "±3"]                 # 시장 모델 학습 대상. 그 외(±1, ±3+)는 0.1% 미만이라 제외
MARKET_CAT_FEATURES = ["work_type", RANGE_COL]
MARKET_NUM_FEATURES = ["est_price_log", "bidder_count", "is_year_end"]
MARKET_FEATURES = MARKET_CAT_FEATURES + MARKET_NUM_FEATURES


def fit_encoders(train: pd.DataFrame) -> dict[str, list[str]]:
    """학습 구간의 범주 목록. RARE_MIN_COUNT 미만 범주는 '기타'로 병합한다."""
    enc: dict[str, list[str]] = {}
    for col, src in (("dept_code", "dept_name"), ("award_method", "award_method")):
        vc = train[src].fillna(OTHER).astype(str).str.strip().value_counts()
        keep = sorted(vc[vc >= RARE_MIN_COUNT].index.tolist())
        enc[col] = keep + [OTHER] if OTHER not in keep else keep
    return enc


def _rolling_pair(rate: pd.Series) -> tuple[pd.Series, pd.Series]:
    """한 계열의 ma/std. 반드시 shift(1)을 먼저 적용해 자기 자신과 미래를 제외한다."""
    r = rate.shift(1)
    return (r.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean(),
            r.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).std(ddof=0))


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """정렬된 df에 ma_15_rate / std_15_rate 추가. 행 i의 값은 i 이전 행들만으로 계산된다.

    SOURCE_COL 이 있으면 source 별로 따로 계산한다(사내 행은 사내 이력만, g2b 행은 g2b 이력만 본다).
    """
    if SOURCE_COL in df.columns and df[SOURCE_COL].nunique(dropna=False) > 1:
        ma = pd.Series(np.nan, index=df.index)
        sd = pd.Series(np.nan, index=df.index)
        for _, idx in df.groupby(SOURCE_COL, sort=False, dropna=False).groups.items():
            m, s = _rolling_pair(df.loc[idx, TARGET])
            ma.loc[idx], sd.loc[idx] = m, s
        df["ma_15_rate"], df["std_15_rate"] = ma, sd
    else:
        df["ma_15_rate"], df["std_15_rate"] = _rolling_pair(df[TARGET])
    return df


def build_features(df: pd.DataFrame, enc: dict[str, list[str]]) -> pd.DataFrame:
    """정제 데이터 → 모델 입력. 반환값은 날짜순 정렬이며 FEATURES + TARGET + DATE_COL 컬럼을 가진다."""
    d = df.sort_values([DATE_COL, "contract_no"], kind="stable").reset_index(drop=True).copy()
    d["work_type"] = d["work_type"].astype(int)
    for col, src in (("dept_code", "dept_name"), ("award_method", "award_method")):
        s = d[src].fillna(OTHER).astype(str).str.strip()
        d[col] = s.where(s.isin(enc[col]), OTHER)
        d[col] = pd.Categorical(d[col], categories=enc[col])
    d["work_type"] = pd.Categorical(d["work_type"], categories=[1, 2])
    if RANGE_COL in d.columns:
        d[RANGE_COL] = pd.Categorical(d[RANGE_COL].astype(object), categories=RANGE_CLASSES)
    d["est_price_log"] = np.log10(d["est_price"].clip(lower=1.0))
    d["bidder_count"] = d["bidder_count"].fillna(1).clip(lower=1).astype(float)
    d["is_year_end"] = d[DATE_COL].dt.month.isin([11, 12]).astype(int)
    d = add_rolling_stats(d)
    return d


def rolling_from_history(history_rates: pd.Series | list[float]) -> tuple[float, float]:
    """추론용: 직전 사정률 이력(시간순)에서 ma_15 / std_15 계산. 이력 부족 시 NaN."""
    s = pd.Series(list(history_rates), dtype=float).tail(ROLL_WINDOW)
    if len(s) < ROLL_MIN_PERIODS:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=0))


def make_inference_row(*, work_type: int, dept_name: str | None, award_method: str | None,
                       est_price: float, bidder_count: float, as_of: pd.Timestamp,
                       history_rates, enc: dict[str, list[str]], range_cls: str | None = None,
                       features: list[str] | None = None) -> pd.DataFrame:
    """CLI 입력 한 건을 features(기본 FEATURES) 순서의 1행 DataFrame으로 만든다 (build_features와 같은 규칙).

    시장 모델은 features=MARKET_FEATURES, range_cls="±3" 처럼 호출한다. 그때 dept/award 인코더는 없어도 된다.
    """
    features = features or FEATURES
    ma, sd = rolling_from_history(history_rates)
    row = pd.DataFrame([{
        "work_type": int(work_type),
        "dept_code": (dept_name or OTHER).strip(),
        "award_method": (award_method or OTHER).strip(),
        RANGE_COL: range_cls,
        "est_price_log": float(np.log10(max(est_price, 1.0))),
        "bidder_count": float(max(bidder_count or 1, 1)),
        "is_year_end": int(pd.Timestamp(as_of).month in (11, 12)),
        "ma_15_rate": ma,
        "std_15_rate": sd,
    }])
    for col in ("dept_code", "award_method"):
        if col in features:
            row[col] = row[col].where(row[col].isin(enc[col]), OTHER)
            row[col] = pd.Categorical(row[col], categories=enc[col])
    row["work_type"] = pd.Categorical(row["work_type"], categories=[1, 2])
    if RANGE_COL in features:
        if range_cls not in RANGE_CLASSES:
            raise ValueError(f"range_cls 는 {RANGE_CLASSES} 중 하나여야 합니다: {range_cls!r}")
        row[RANGE_COL] = pd.Categorical(row[RANGE_COL], categories=RANGE_CLASSES)
    return row[features]
