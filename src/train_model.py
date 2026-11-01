"""분위수 회귀 학습 + 5-Fold 시계열 교차검증 (Phase 3).

- LightGBM Quantile(α=0.25/0.5/0.75), 과적합 통제 하이퍼파라미터는 CLAUDE.md 상한을 따른다.
- TimeSeriesSplit(5). 폴드마다 인코더·학습평균을 학습 구간에서만 적합한다.
- 조기종료는 학습 구간 끝 15%를 홀드아웃으로 써서 검증 폴드를 보지 않는다.
- 최종 예측 = 0.7*ML + 0.3*사전값 (베이지안 축소). 사전값은 Q50은 학습평균, Q25/Q75는 학습 구간의
  경험적 25/75 분위수다. (평균 하나로 축소하면 구간 폭이 30% 좁아져 커버리지가 무너진다.)
- `--with-g2b`: 나라장터 시장 데이터(data/g2b_detail_all.parquet)를 사내 데이터와 합쳐 학습한다(Phase 3-6 실험 3b).
  g2b 행은 dept_code="시장", award_method="기타", est_price=base_amt/1.1, 날짜는 open_dt 를 contract_date 로 맞춘다.
  source 피처는 쓰지 않고 sample_weight(사내 3.0, g2b 1.0)로 합친다. 이동통계는 features.py 가 source 별로 따로 낸다.
  TimeSeriesSplit 폴드는 사내 행에서만 나누고, 각 폴드의 g2b 학습행은 사내 학습구간 마지막 날짜 이전 것만 쓴다.
  Baseline·사전값·검증 지표는 모두 사내 행 기준이라 3a(사내 단독)와 직접 비교된다.
- `--compare`: 3a/3b 를 같은 폴드에서 돌려 out/cv_compare.md 를 쓴다(모델 저장 없음).
- 수의계약(contract_method가 "수의"로 시작)은 학습·검증에서 분리한다. 기초금액이 있어 Phase 1 필터를
  통과하지만 예정가격이 15개 예가 추첨으로 정해지는 경쟁입찰과 절차가 달라 사정률 분포가 전혀 다르다
  (하방으로 치우치고 분산이 훨씬 크다). 모델은 경쟁입찰 전용이고, 수의계약 사정률은 경험 분포
  (건수·평균·분위수)로 아티팩트에 저장해 predict_cli --negotiated 가 쓴다.
- `--source market` (Phase 6): 나라장터 시장 데이터(data/g2b_detail_all.parquet, 예가 범위 ±2%/±3% 건)로 공공발주 전체용
  시장 모델을 학습한다. 피처는 features.MARKET_FEATURES, 사전값·Baseline 은 예가 범위(range_cls)별로 둔다.
  출력: out/cv_report_market.md, models/market_model.joblib. predict_cli --market 이 쓴다.
출력: out/cv_report.md, models/bidding_model.joblib
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "internal_clean_post2019.parquet"
G2B_PATH = ROOT / "data" / "g2b_detail_all.parquet"
MODEL_PATH = ROOT / "models" / "bidding_model.joblib"
REPORT_PATH = ROOT / "out" / "cv_report.md"
MARKET_MODEL_PATH = ROOT / "models" / "market_model.joblib"
MARKET_REPORT_PATH = ROOT / "out" / "cv_report_market.md"
COMPARE_PATH = ROOT / "out" / "cv_compare.md"

W_INTERNAL, W_G2B = 3.0, 1.0        # --with-g2b 샘플 가중치 (작업계획 Phase 3-6)
G2B_DEPT, G2B_AWARD = "시장", "기타"
RATE_LO, RATE_HI = 96.5, 103.5      # 사내와 같은 사정률 학습 필터
EST_PRICE_DIV = 1.1                 # 기초금액(부가세 포함) → 추정가격: 사내 규칙과 동일한 1.1 배
WEIGHT_COL = "sample_weight"
MIN_GAIN = 0.002                    # 3b 채택 최소 개선폭(%p): 사내 검증 MAE 평균이 이만큼은 낮아야 한다

NEGOTIATED_METHOD = "수의"          # contract_method 가 이 문자열로 시작하면 수의계약 → 학습 제외, 경험 분포로 별도 저장
NEG_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

QUANTILES = (0.25, 0.50, 0.75)
SHRINK_ML = 0.7            # 최종 = 0.7*ML + 0.3*사전값(Q50: 학습평균, Q25/Q75: 학습 분위수)
EARLY_STOP_FRAC = 0.15     # 학습 구간 끝 15%를 조기종료 홀드아웃으로
N_SPLITS = 5

LGB_PARAMS = dict(
    objective="quantile",
    max_depth=3,
    num_leaves=7,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    learning_rate=0.03,
    n_estimators=300,
    reg_lambda=1.0,
    min_split_gain=0.0,
    verbose=-1,
    random_state=42,
)


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def fit_quantile_models(X: pd.DataFrame, y: pd.Series, n_estimators: dict[float, int] | None = None,
                        w: pd.Series | None = None, cat_features: list[str] | None = None):
    """α별 LightGBM 적합. n_estimators가 없으면 학습 구간 끝 15%로 조기종료해 반복수를 정한다.

    w 가 있으면 sample_weight 로 쓴다(조기종료 홀드아웃에도 같은 가중치 적용). X 는 날짜순이어야 한다.
    """
    models, best = {}, {}
    cat_features = F.CAT_FEATURES if cat_features is None else cat_features
    n = len(X)
    cut = int(n * (1 - EARLY_STOP_FRAC))
    w_arr = None if w is None else np.asarray(w, dtype=float)
    for a in QUANTILES:
        params = {**LGB_PARAMS, "alpha": a}
        if n_estimators and a in n_estimators:
            m = lgb.LGBMRegressor(**{**params, "n_estimators": n_estimators[a]})
            m.fit(X, y, sample_weight=w_arr, categorical_feature=cat_features)
            best[a] = n_estimators[a]
        else:
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[:cut], y.iloc[:cut], eval_X=X.iloc[cut:], eval_y=y.iloc[cut:],
                  sample_weight=None if w_arr is None else w_arr[:cut],
                  eval_sample_weight=None if w_arr is None else [w_arr[cut:]],
                  categorical_feature=cat_features,
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            best[a] = int(m.best_iteration_ or params["n_estimators"])
            # 홀드아웃까지 포함해 같은 반복수로 재적합
            m = lgb.LGBMRegressor(**{**params, "n_estimators": best[a]})
            m.fit(X, y, sample_weight=w_arr, categorical_feature=cat_features)
        models[a] = m
    return models, best


def make_priors(y_train: pd.Series) -> dict[float, float]:
    """축소 대상 사전값: Q50은 학습평균, Q25/Q75는 학습 구간 경험적 분위수."""
    return {0.25: float(y_train.quantile(0.25)), 0.50: float(y_train.mean()), 0.75: float(y_train.quantile(0.75))}


def make_group_priors(y_train: pd.Series, groups: pd.Series) -> dict[str, dict[float, float]]:
    """그룹(예: 예가 범위)별 사전값. 시장 모델은 ±2%/±3% 의 분포가 달라 사전값을 범위별로 둔다."""
    return {str(g): make_priors(y_train[groups.values == g]) for g in pd.unique(groups.values)}


def priors_for_rows(group_priors: dict[str, dict[float, float]], groups) -> dict[float, np.ndarray]:
    """행별 그룹에 맞는 사전값 배열. predict_quantiles 의 priors 로 그대로 쓴다."""
    g = np.asarray([str(x) for x in groups])
    return {a: np.array([group_priors[x][a] for x in g]) for a in QUANTILES}


def predict_quantiles(models: dict, X: pd.DataFrame, priors: dict, shrink: float = SHRINK_ML) -> dict[float, np.ndarray]:
    """priors[a] 는 스칼라(사내 모델) 또는 행별 배열(시장 모델: 범위별 사전값)."""
    out = {}
    for a, m in models.items():
        raw = m.predict(X)
        out[a] = shrink * raw + (1 - shrink) * np.asarray(priors[a], dtype=float)
    # 분위수 교차 방지
    out[0.25] = np.minimum(out[0.25], out[0.50])
    out[0.75] = np.maximum(out[0.75], out[0.50])
    return out


def g2b_to_internal_schema(g: pd.DataFrame) -> pd.DataFrame:
    """나라장터 빌드 parquet(g2b_detail_*.parquet) → 사내 정제본과 같은 컬럼의 학습용 행.

    - 사정률 필터 RATE_LO~RATE_HI, base_amt>0, open_dt 결측 제외 (사내 학습 필터와 동일 취지)
    - contract_date := open_dt (공통 날짜 컬럼), contract_no := bid_key (정렬 타이브레이커)
    - dept_name="시장", award_method="기타", est_price=base_amt/1.1, source="g2b"
    """
    g = g.copy()
    g["open_dt"] = pd.to_datetime(g["open_dt"], errors="coerce")
    keep = (g["base_amt"] > 0) & g["yega_rate"].between(RATE_LO, RATE_HI) & g["open_dt"].notna()
    g = g[keep]
    return pd.DataFrame({
        "contract_no": g["bid_key"].astype(str).values,
        F.DATE_COL: g["open_dt"].values,
        "dept_name": G2B_DEPT,
        "work_type": g["work_type"].astype(int).values,
        "award_method": G2B_AWARD,
        "est_price": (g["base_amt"] / EST_PRICE_DIV).astype(float).values,
        "bidder_count": pd.to_numeric(g["bidder_count"], errors="coerce").values,
        F.TARGET: g[F.TARGET].astype(float).values,
        F.SOURCE_COL: "g2b",
    })


def load_g2b(path: Path = G2B_PATH) -> pd.DataFrame:
    return g2b_to_internal_schema(pd.read_parquet(path))


def market_schema(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """나라장터 빌드 parquet → 시장 모델 학습 행 (Phase 6).

    - 예가 범위 range_cls 가 RANGE_CLASSES(±2/±3) 인 건만 쓴다(그 외 0.1% 미만). 사정률 필터·base_amt>0·open_dt 결측 제외.
    - 참여업체수 결측(목록 미조인 약 8%)은 시장 중앙값으로 채운다(사내 규칙의 "결측=1"은 수의계약 의미라 시장엔 맞지 않음).
    - dept_name/award_method 는 상수("시장"/"기타")로 두어 features.build_features 가 그대로 돈다(시장 피처엔 안 들어감).
    """
    g = g.copy()
    g["open_dt"] = pd.to_datetime(g["open_dt"], errors="coerce")
    n0 = len(g)
    keep = ((g["base_amt"] > 0) & g["yega_rate"].between(RATE_LO, RATE_HI) & g["open_dt"].notna()
            & g[F.RANGE_COL].astype(object).isin(F.RANGE_CLASSES))
    g = g[keep]
    bidders = pd.to_numeric(g["bidder_count"], errors="coerce")
    bidder_median = float(bidders.median()) if bidders.notna().any() else 1.0
    df = pd.DataFrame({
        "contract_no": g["bid_key"].astype(str).values,
        F.DATE_COL: g["open_dt"].values,
        "dept_name": G2B_DEPT,
        "work_type": g["work_type"].astype(int).values,
        "award_method": G2B_AWARD,
        F.RANGE_COL: g[F.RANGE_COL].astype(object).values,
        "est_price": (g["base_amt"] / EST_PRICE_DIV).astype(float).values,
        "bidder_count": bidders.fillna(bidder_median).values,
        F.TARGET: g[F.TARGET].astype(float).values,
        F.SOURCE_COL: "g2b",
    })
    meta = {"n_raw": int(n0), "n_used": int(len(df)), "bidder_median": bidder_median,
            "bidder_missing_frac": float(bidders.isna().mean()) if len(bidders) else 0.0}
    return df.sort_values([F.DATE_COL, "contract_no"], kind="stable").reset_index(drop=True), meta


def load_market(path: Path = G2B_PATH) -> tuple[pd.DataFrame, dict]:
    return market_schema(pd.read_parquet(path))


def range_stats(df: pd.DataFrame) -> pd.DataFrame:
    """예가 범위별 사정률 통계(리포트·아티팩트용)."""
    g = df.groupby(F.RANGE_COL, observed=True)[F.TARGET]
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "std": g.std(ddof=0),
                         "q25": g.quantile(0.25), "q50": g.quantile(0.5), "q75": g.quantile(0.75)})


def split_negotiated(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(경쟁입찰 행, 수의계약 행). contract_method 컬럼이 없으면 전부 경쟁입찰로 본다."""
    if "contract_method" not in df.columns:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    is_neg = df["contract_method"].fillna("").astype(str).str.strip().str.startswith(NEGOTIATED_METHOD)
    return df[~is_neg].reset_index(drop=True), df[is_neg].reset_index(drop=True)


def negotiated_distribution(neg: pd.DataFrame) -> dict | None:
    """수의계약 사정률의 경험 분포. predict_cli --negotiated 가 Q25/Q50/Q75 로 쓴다. 행이 없으면 None."""
    if neg is None or neg.empty:
        return None
    y = neg[F.TARGET].astype(float)
    out = {
        "n": int(len(y)), "mean": float(y.mean()), "std": float(y.std(ddof=0)),
        "quantiles": {a: float(y.quantile(a)) for a in NEG_QUANTILES},
        "by_work_type": {},
    }
    if "work_type" in neg.columns:
        for wt, g in y.groupby(neg["work_type"].astype(int)):
            out["by_work_type"][int(wt)] = {"n": int(len(g)), "mean": float(g.mean()),
                                            "quantiles": {a: float(g.quantile(a)) for a in NEG_QUANTILES}}
    if F.DATE_COL in neg.columns:
        d = pd.to_datetime(neg[F.DATE_COL])
        out["date_range"] = [str(d.min().date()), str(d.max().date())]
    return out


def _prep_internal(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d[F.SOURCE_COL] = "internal"
    d[WEIGHT_COL] = W_INTERNAL
    return d.sort_values([F.DATE_COL, "contract_no"], kind="stable").reset_index(drop=True)


def _prep_extra(extra: pd.DataFrame | None, w_extra: float) -> pd.DataFrame | None:
    if extra is None or extra.empty:
        return None
    e = extra.copy()
    e[WEIGHT_COL] = w_extra
    return e.sort_values([F.DATE_COL, "contract_no"], kind="stable").reset_index(drop=True)


def _norm_weights(w: pd.Series, w_internal: float) -> pd.Series:
    """가중치를 사내 행이 1.0 이 되도록 나눈다(비율 3:1 유지). LightGBM 의 reg_lambda·min_sum_hessian 은
    가중치의 절대 크기에 반응하므로, 이렇게 해야 사내 행의 정규화 조건이 3a(가중치 없음)와 같아진다."""
    return w / w_internal


def run_cv(df: pd.DataFrame, extra: pd.DataFrame | None = None,
           w_internal: float = W_INTERNAL, w_extra: float = W_G2B,
           features: list[str] | None = None, cat_features: list[str] | None = None,
           prior_group_col: str | None = None) -> tuple[pd.DataFrame, dict]:
    """TimeSeriesSplit 교차검증. 폴드별 성적 DataFrame과 평균 best_iteration 반환.

    extra(g2b 행)가 있으면 폴드마다 사내 학습구간 마지막 날짜 이전의 extra 행을 학습에만 보탠다.
    폴드 분할·Baseline·사전값·검증/학습 MAE·커버리지는 모두 사내 행만으로 계산해 3a/3b 가 직접 비교된다.
    features/cat_features 기본은 사내 모델(F.FEATURES). prior_group_col 이 있으면(시장 모델: range_cls) 사전값과 Baseline 을
    그 그룹별로 둔다.
    """
    features = F.FEATURES if features is None else features
    cat_features = F.CAT_FEATURES if cat_features is None else cat_features
    df = _prep_internal(df)
    df[WEIGHT_COL] = w_internal
    extra = _prep_extra(extra, w_extra)
    df["_rid"] = np.arange(len(df))
    if extra is not None:
        extra["_rid"] = np.arange(len(df), len(df) + len(extra))
    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rows, iters = [], {a: [] for a in QUANTILES}
    for k, (tr, va) in enumerate(tss.split(df), start=1):
        train, val = df.iloc[tr], df.iloc[va]
        cutoff = train[F.DATE_COL].max()
        ex_tr = extra[extra[F.DATE_COL] <= cutoff] if extra is not None else None
        if ex_tr is not None and ex_tr.empty:
            ex_tr = None                                           # 시장 행이 없는 폴드는 3a 와 완전히 같게
        train_all = train if ex_tr is None else pd.concat([train, ex_tr], ignore_index=True)
        enc = F.fit_encoders(train_all)
        # 피처는 전체 시계열에서 한 번에 만들되(이동통계는 source 별 과거만 참조), 인코더는 학습구간 것만 쓴다
        pool = df if ex_tr is None else pd.concat([df, ex_tr], ignore_index=True)
        feat = F.build_features(pool, enc)                      # 날짜순 정렬됨
        in_tr = feat["_rid"].isin(train_all["_rid"]).values
        in_va = feat["_rid"].isin(val["_rid"]).values
        is_int = (feat[F.SOURCE_COL] == "internal").values
        Xtr, ytr = feat.loc[in_tr, features], feat.loc[in_tr, F.TARGET]
        wtr = _norm_weights(feat.loc[in_tr, WEIGHT_COL], w_internal) if ex_tr is not None else None
        Xva, yva = feat.loc[in_va, features], feat.loc[in_va, F.TARGET]
        # 학습 지표(Gap)·Baseline·사전값은 사내 학습행 기준
        Xtr_i, ytr_i = feat.loc[in_tr & is_int, features], feat.loc[in_tr & is_int, F.TARGET]
        train_mean = float(ytr_i.mean())
        if prior_group_col:
            gp = make_group_priors(ytr_i, feat.loc[in_tr & is_int, prior_group_col].astype(str))
            priors_tr = priors_for_rows(gp, feat.loc[in_tr & is_int, prior_group_col])
            priors_va = priors_for_rows(gp, feat.loc[in_va, prior_group_col])
            base_tr, base_va = priors_tr[0.5], priors_va[0.5]            # Baseline = 그룹 평균
            priors_cov = priors_va
        else:
            priors_tr = priors_va = make_priors(ytr_i)
            base_tr, base_va = train_mean, train_mean
            priors_cov = {a: np.full(len(yva), priors_va[a]) for a in QUANTILES}

        models, best = fit_quantile_models(Xtr, ytr, w=wtr, cat_features=cat_features)
        for a in QUANTILES:
            iters[a].append(best[a])

        ytr = ytr_i
        p_tr_raw = {a: m.predict(Xtr_i) for a, m in models.items()}
        p_va_raw = {a: m.predict(Xva) for a, m in models.items()}
        p_tr = predict_quantiles(models, Xtr_i, priors_tr)
        p_va = predict_quantiles(models, Xva, priors_va)

        base_mae_va = float(np.mean(np.abs(yva - base_va)))
        base_mae_tr = float(np.mean(np.abs(ytr - base_tr)))
        mae_tr_raw = float(np.mean(np.abs(ytr - p_tr_raw[0.5])))
        mae_va_raw = float(np.mean(np.abs(yva - p_va_raw[0.5])))
        mae_tr = float(np.mean(np.abs(ytr - p_tr[0.5])))
        mae_va = float(np.mean(np.abs(yva - p_va[0.5])))
        cov = float(np.mean((yva >= p_va[0.25]) & (yva <= p_va[0.75])))
        # 기준별 투찰가 통과율: Qα 기준 투찰가는 실제 사정률이 Qα 이하일 때만 낙찰하한을 넘는다(안전마진 0 가정)
        pass_rate = {a: float(np.mean(yva <= p_va[a])) for a in QUANTILES}
        cov_raw = float(np.mean((yva >= p_va_raw[0.25]) & (yva <= p_va_raw[0.75])))
        cov_prior = float(np.mean((yva >= priors_cov[0.25]) & (yva <= priors_cov[0.75])))
        rows.append({
            "fold": k,
            "train_n": len(tr), "val_n": len(va),
            "extra_n": 0 if ex_tr is None else int(len(ex_tr)),
            "train_period": f"{train[F.DATE_COL].min():%Y-%m}~{train[F.DATE_COL].max():%Y-%m}",
            "val_period": f"{val[F.DATE_COL].min():%Y-%m}~{val[F.DATE_COL].max():%Y-%m}",
            "train_mean": train_mean,
            "baseline_mae_val": base_mae_va,
            "baseline_mae_train": base_mae_tr,
            "ml_mae_train_raw": mae_tr_raw, "ml_mae_val_raw": mae_va_raw,
            "gap_raw": mae_va_raw - mae_tr_raw,
            "ml_mae_train": mae_tr, "ml_mae_val": mae_va,
            "gap": mae_va - mae_tr,
            "improve_pct": (base_mae_va - mae_va) / base_mae_va * 100,
            "coverage_q25_q75": cov,
            "coverage_raw": cov_raw,
            "coverage_prior": cov_prior,
            "pass_q25": pass_rate[0.25], "pass_q50": pass_rate[0.50], "pass_q75": pass_rate[0.75],
            "pinball_q25": pinball(yva.values, p_va[0.25], 0.25),
            "pinball_q50": pinball(yva.values, p_va[0.50], 0.50),
            "pinball_q75": pinball(yva.values, p_va[0.75], 0.75),
            "best_iter_q50": best[0.5],
        })
    cv = pd.DataFrame(rows)
    avg_iters = {a: max(20, int(np.mean(v))) for a, v in iters.items()}
    return cv, avg_iters


def feature_importance(models: dict, features: list[str] | None = None) -> pd.DataFrame:
    imp = pd.DataFrame({f"q{int(a*100)}": m.booster_.feature_importance("gain") for a, m in models.items()},
                       index=F.FEATURES if features is None else features)
    imp = imp.div(imp.sum(axis=0), axis=1).round(3)
    return imp.sort_values("q50", ascending=False)


def write_report(cv: pd.DataFrame, imp: pd.DataFrame, n_total: int, avg_iters: dict, path: Path,
                 n_extra: int = 0, neg_dist: dict | None = None, data_desc: str | None = None,
                 title: str = "5-Fold 시계열 교차검증 성적표", rstats: pd.DataFrame | None = None) -> str:
    m = cv.mean(numeric_only=True)
    gap_ok = m["gap"] <= 0.08
    cov_ok = 0.45 <= m["coverage_q25_q75"] <= 0.55
    if data_desc is None:
        data_desc = f"사내 경쟁입찰 {n_total:,}건"
        if neg_dist:
            data_desc += f" (수의계약 {neg_dist['n']}건은 학습 제외)"
    if n_extra:
        data_desc += f" + 나라장터 {n_extra:,}건 (가중치 사내 {W_INTERNAL:g} : 시장 {W_G2B:g}, 검증은 사내 행만)"
    lines = [
        f"# {title}",
        "",
        f"- 생성: {dt.datetime.now():%Y-%m-%d %H:%M} | 데이터: {data_desc} | 모델: LightGBM Quantile(Q25/Q50/Q75)",
        f"- 하이퍼파라미터: max_depth={LGB_PARAMS['max_depth']}, num_leaves={LGB_PARAMS['num_leaves']}, min_child_samples={LGB_PARAMS['min_child_samples']}, subsample={LGB_PARAMS['subsample']}, lr={LGB_PARAMS['learning_rate']}, 반복수(조기종료 평균) Q25/Q50/Q75 = {avg_iters[0.25]}/{avg_iters[0.5]}/{avg_iters[0.75]}",
        f"- 최종 예측 = {SHRINK_ML}×ML + {1-SHRINK_ML:.1f}×사전값 (Q50: 학습평균, Q25/Q75: 학습 구간 경험적 분위수)",
        "",
        "## 폴드별 (단위: %p)",
        "",
        "| fold | 학습기간 | 검증기간 | n학습/n검증 | Baseline MAE | ML MAE(학습) | ML MAE(검증) | Gap | 개선율 | Q25~Q75 커버리지 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in cv.iterrows():
        lines.append(f"| {int(r.fold)} | {r.train_period} | {r.val_period} | {int(r.train_n)}/{int(r.val_n)} | {r.baseline_mae_val:.4f} | {r.ml_mae_train:.4f} | {r.ml_mae_val:.4f} | {r.gap:+.4f} | {r.improve_pct:+.2f}% | {r.coverage_q25_q75*100:.1f}% |")
    lines += [
        f"| **평균** | | | | **{m['baseline_mae_val']:.4f}** | **{m['ml_mae_train']:.4f}** | **{m['ml_mae_val']:.4f}** | **{m['gap']:+.4f}** | **{m['improve_pct']:+.2f}%** | **{m['coverage_q25_q75']*100:.1f}%** |",
        "",
        "## 판정",
        "",
        f"- Overfit Gap(검증 MAE − 학습 MAE) 평균 {m['gap']:+.4f}%p → 기준 0.08%p 이내: {'통과' if gap_ok else '미달'}",
        f"- 축소 블렌딩 전(ML 원시) Gap 평균 {m['gap_raw']:+.4f}%p, 검증 MAE {m['ml_mae_val_raw']:.4f} (블렌딩 후 {m['ml_mae_val']:.4f})",
        f"- Baseline(학습평균) 대비 검증 MAE 개선율 평균 {m['improve_pct']:+.2f}%",
        f"- Q25~Q75 커버리지 평균 {m['coverage_q25_q75']*100:.1f}% → 목표 45~55%: {'통과' if cov_ok else '미달'}"
        f" (ML 원시 {m['coverage_raw']*100:.1f}%, 사전 분위수만 {m['coverage_prior']*100:.1f}%)",
        f"- Pinball loss 평균 Q25 {m['pinball_q25']:.4f} / Q50 {m['pinball_q50']:.4f} / Q75 {m['pinball_q75']:.4f}",
        f"- **기준별 투찰가 통과율**(실제 사정률 ≤ Qα 비율, 안전마진 0): Q25 {m['pass_q25']*100:.1f}% / Q50 {m['pass_q50']*100:.1f}% / Q75 {m['pass_q75']*100:.1f}%"
        " → 낙찰하한금액은 실제 예정가격에 비례하므로 Qα 기준 투찰가는 실제 사정률이 Qα 이하일 때만 유효하다. 안전한 기준은 Q75다.",
        "",
        "## 피처 중요도 (gain 비율, 최종 모델)",
        "",
        imp.to_markdown(),
        "",
    ]
    if rstats is not None:
        lines += ["## 예가 범위별 사정률 (학습 데이터)", "",
                  "| 범위 | n | mean | std | Q25 | Q50 | Q75 |", "|---|---|---|---|---|---|---|"]
        for r_, row in rstats.iterrows():
            lines.append(f"| {r_} | {int(row['n']):,} | {row['mean']:.4f} | {row['std']:.4f} | {row['q25']:.3f} | {row['q50']:.3f} | {row['q75']:.3f} |")
        lines += ["", "- 사정률 분포를 결정하는 것은 예가 범위다. 공사/용역·참여업체수·기관 유형·규모·월은 평균 0.1%p, std 0.05 이내의 차이만 낸다.", ""]
    if neg_dist:
        q = neg_dist["quantiles"]
        lines += [
            "## 수의계약 (학습 제외, 경험 분포 → predict_cli --negotiated)",
            "",
            f"- {neg_dist['n']}건 ({neg_dist.get('date_range', ['?', '?'])[0]} ~ {neg_dist.get('date_range', ['?', '?'])[1]}) | "
            f"사정률 mean {neg_dist['mean']:.3f} | std {neg_dist['std']:.3f} | "
            f"Q05 {q[0.05]:.3f} / Q25 {q[0.25]:.3f} / Q50 {q[0.5]:.3f} / Q75 {q[0.75]:.3f} / Q95 {q[0.95]:.3f}",
            "- 경쟁입찰(15개 예가 중 4개 추첨)과 절차가 달라 분포가 넓고 하방으로 치우친다. ML 모델은 이 행을 보지 않는다.",
            "",
        ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def compare_verdict(cv_a: pd.DataFrame, cv_b: pd.DataFrame) -> tuple[bool, str]:
    """3b 채택 여부: 사내 검증 MAE 평균이 낮고, Gap·커버리지 기준을 유지해야 한다."""
    a, b = cv_a.mean(numeric_only=True), cv_b.mean(numeric_only=True)
    wins = int((cv_b["ml_mae_val"] < cv_a["ml_mae_val"]).sum())
    diff = b["ml_mae_val"] - a["ml_mae_val"]
    better_mae = (diff <= -MIN_GAIN) and (wins * 2 > len(cv_a))     # 최소 개선폭 + 폴드 과반 개선
    gap_ok = b["gap"] <= 0.08
    cov_ok = 0.45 <= b["coverage_q25_q75"] <= 0.55
    reasons = [f"검증 MAE 평균 3a {a['ml_mae_val']:.4f} → 3b {b['ml_mae_val']:.4f} ({diff:+.4f}%p, 폴드 {wins}/{len(cv_a)} 개선; "
               f"채택 조건 {-MIN_GAIN:+.3f}%p 이하 & 과반 폴드 개선: {'충족' if better_mae else '미충족'})",
               f"3b Gap {b['gap']:+.4f}%p ({'통과' if gap_ok else '기준 0.08 초과'})",
               f"3b 커버리지 {b['coverage_q25_q75']*100:.1f}% ({'통과' if cov_ok else '45~55% 이탈'})"]
    adopt = bool(better_mae and gap_ok and cov_ok)
    return adopt, "; ".join(reasons)


def write_compare(cv_a: pd.DataFrame, cv_b: pd.DataFrame, n_int: int, n_g2b: int, g2b_range: tuple[str, str],
                  path: Path) -> str:
    """실험 3a(사내 단독) vs 3b(사내+시장) 폴드별 비교표와 결론."""
    adopt, reason = compare_verdict(cv_a, cv_b)
    a, b = cv_a.mean(numeric_only=True), cv_b.mean(numeric_only=True)
    lines = [
        "# 실험 3a(사내 단독) vs 3b(사내+나라장터 시장) 비교",
        "",
        f"- 생성: {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- 사내 경쟁입찰 {n_int:,}건(수의계약 제외) | 나라장터 학습 후보 {n_g2b:,}건 (open_dt {g2b_range[0]} ~ {g2b_range[1]}, 사정률 {RATE_LO}~{RATE_HI}% 필터 후)",
        f"- 3b 설정: dept_code=\"{G2B_DEPT}\", award_method=\"{G2B_AWARD}\", est_price=base_amt/{EST_PRICE_DIV}, source 피처 없음, "
        f"sample_weight 사내 {W_INTERNAL:g} / 시장 {W_G2B:g}, 이동통계는 source 별 분리 계산",
        "- 폴드는 사내 행만으로 TimeSeriesSplit(5). 각 폴드의 시장 행은 사내 학습구간 마지막 날짜 이전 것만 학습에 추가(`시장n`).",
        "- Baseline·사전값·모든 지표는 사내 행 기준이라 두 실험의 Baseline MAE 는 동일하다.",
        f"- 가중치는 사내 행이 1.0 이 되도록 정규화해 넣는다(시장 {W_G2B / W_INTERNAL:.3f}). 시장 행이 없는 폴드는 3a 와 수치가 같다.",
        "",
        "## 폴드별 (단위: %p)",
        "",
        "| fold | 검증기간 | 사내n | 시장n(3b) | Baseline MAE | 3a ML MAE | 3b ML MAE | Δ(3b−3a) | 3a Gap | 3b Gap | 3a 커버리지 | 3b 커버리지 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (_, ra), (_, rb) in zip(cv_a.iterrows(), cv_b.iterrows()):
        lines.append(
            f"| {int(ra.fold)} | {ra.val_period} | {int(ra.train_n)}/{int(ra.val_n)} | {int(rb.extra_n):,} | {ra.baseline_mae_val:.4f} "
            f"| {ra.ml_mae_val:.4f} | {rb.ml_mae_val:.4f} | {rb.ml_mae_val - ra.ml_mae_val:+.4f} "
            f"| {ra.gap:+.4f} | {rb.gap:+.4f} | {ra.coverage_q25_q75*100:.1f}% | {rb.coverage_q25_q75*100:.1f}% |")
    lines += [
        f"| **평균** | | | | **{a['baseline_mae_val']:.4f}** | **{a['ml_mae_val']:.4f}** | **{b['ml_mae_val']:.4f}** "
        f"| **{b['ml_mae_val'] - a['ml_mae_val']:+.4f}** | **{a['gap']:+.4f}** | **{b['gap']:+.4f}** "
        f"| **{a['coverage_q25_q75']*100:.1f}%** | **{b['coverage_q25_q75']*100:.1f}%** |",
        "",
        "## 보조 지표 (평균)",
        "",
        "| 실험 | Baseline 대비 개선율 | ML 원시 검증 MAE | 원시 Gap | Pinball Q25/Q50/Q75 | 반복수 Q50 |",
        "|---|---|---|---|---|---|",
        f"| 3a 사내 단독 | {a['improve_pct']:+.2f}% | {a['ml_mae_val_raw']:.4f} | {a['gap_raw']:+.4f} | {a['pinball_q25']:.4f}/{a['pinball_q50']:.4f}/{a['pinball_q75']:.4f} | {a['best_iter_q50']:.0f} |",
        f"| 3b 사내+시장 | {b['improve_pct']:+.2f}% | {b['ml_mae_val_raw']:.4f} | {b['gap_raw']:+.4f} | {b['pinball_q25']:.4f}/{b['pinball_q50']:.4f}/{b['pinball_q75']:.4f} | {b['best_iter_q50']:.0f} |",
        "",
        "## 결론",
        "",
        f"- {reason}",
        f"- **{'3b 채택: 시장 데이터가 사내 검증 MAE 를 개선한다. 기본값을 --with-g2b 로 바꿀 수 있다.' if adopt else '3a 유지: 시장 데이터가 사내 검증 MAE 를 개선하지 못했다. 기본값(사내 단독)을 유지한다.'}**",
        "",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def train_market(args: argparse.Namespace) -> int:
    """Phase 6: 나라장터 시장 모델. 피처 MARKET_FEATURES, 사전값·Baseline 은 예가 범위별."""
    model_out = args.model_out if args.model_out != MODEL_PATH else MARKET_MODEL_PATH
    report_out = args.report_out if args.report_out != REPORT_PATH else MARKET_REPORT_PATH
    df, meta = load_market(args.g2b_data)
    rs = range_stats(df)
    print(f"나라장터 {meta['n_raw']:,}건 → 학습 {meta['n_used']:,}건 (예가 범위 ±2/±3, 사정률 {RATE_LO}~{RATE_HI}) "
          f"| 참여업체수 결측 {meta['bidder_missing_frac']*100:.1f}% → 중앙값 {meta['bidder_median']:.0f} 대체")
    print(rs.round(4).to_string())

    cv, avg_iters = run_cv(df, features=F.MARKET_FEATURES, cat_features=F.MARKET_CAT_FEATURES, prior_group_col=F.RANGE_COL)

    full = _prep_internal(df)
    enc = F.fit_encoders(full)
    feat = F.build_features(full, enc)
    X, y = feat[F.MARKET_FEATURES], feat[F.TARGET]
    models, _ = fit_quantile_models(X, y, n_estimators=avg_iters, cat_features=F.MARKET_CAT_FEATURES)
    imp = feature_importance(models, F.MARKET_FEATURES)
    gp = make_group_priors(y, feat[F.RANGE_COL].astype(str))

    desc = (f"나라장터 시장 {len(df):,}건 (예가 범위 " + " / ".join(f"{k} {int(v):,}" for k, v in rs['n'].items())
            + f"; 개찰 {df[F.DATE_COL].min():%Y-%m}~{df[F.DATE_COL].max():%Y-%m})")
    text = write_report(cv, imp, len(df), avg_iters, report_out, data_desc=desc,
                        title="시장 모델 5-Fold 시계열 교차검증 성적표 (Phase 6)", rstats=rs)
    print(text)

    if not args.no_save:
        artifact = {
            "source": "market",
            "models": models,
            "features": F.MARKET_FEATURES,
            "cat_features": F.MARKET_CAT_FEATURES,
            "encoders": enc,
            "range_classes": F.RANGE_CLASSES,
            "priors_by_range": gp,                       # predict_cli --market 이 --range 에 맞는 사전값을 쓴다
            "priors": gp[max(gp, key=lambda k: rs.loc[k, "n"])],   # 호환용: 최다 범위(±3)의 사전값
            "train_mean": float(y.mean()),
            "train_std": float(y.std(ddof=0)),
            "range_stats": rs.to_dict("index"),
            "bidder_median": meta["bidder_median"],
            "shrink_ml": SHRINK_ML,
            "n_estimators": avg_iters,
            "history_tail": None,                        # 시장 모델은 이동통계를 쓰지 않는다
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "n_train": int(len(df)),
            "n_negotiated": 0, "negotiated_dist": None,
            "data_range": [str(df[F.DATE_COL].min().date()), str(df[F.DATE_COL].max().date())],
            "cv_summary": cv.mean(numeric_only=True).to_dict(),
            "lgb_params": LGB_PARAMS,
        }
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_out)
        print(f"\n시장 모델 저장: {model_out} | 리포트: {report_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="분위수 모델 학습 + 시계열 CV")
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--model-out", type=Path, default=MODEL_PATH)
    ap.add_argument("--report-out", type=Path, default=REPORT_PATH)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--with-g2b", action="store_true", help="나라장터 시장 데이터를 가중치로 합쳐 학습 (실험 3b)")
    ap.add_argument("--g2b-data", type=Path, default=G2B_PATH)
    ap.add_argument("--compare", action="store_true", help="3a/3b 를 같은 폴드에서 비교해 --compare-out 에 기록 (모델 저장 없음)")
    ap.add_argument("--compare-out", type=Path, default=COMPARE_PATH)
    ap.add_argument("--source", choices=["internal", "market"], default="internal",
                    help="internal=사내 경쟁입찰 모델(기본), market=나라장터 시장 모델(Phase 6, 공공발주 전체용)")
    args = ap.parse_args(argv)

    if args.source == "market":
        return train_market(args)

    raw = pd.read_parquet(args.data)
    raw[F.DATE_COL] = pd.to_datetime(raw[F.DATE_COL])
    df, neg = split_negotiated(raw)
    neg_dist = negotiated_distribution(neg)
    print(f"데이터 {len(raw):,}건 ({raw[F.DATE_COL].min():%Y-%m-%d} ~ {raw[F.DATE_COL].max():%Y-%m-%d}) "
          f"→ 경쟁입찰 {len(df):,}건 학습, 수의계약 {len(neg):,}건 분리(경험 분포)")
    if neg_dist:
        print(f"  수의계약 사정률 mean {neg_dist['mean']:.3f} std {neg_dist['std']:.3f} "
              f"Q25/Q50/Q75 {neg_dist['quantiles'][0.25]:.3f}/{neg_dist['quantiles'][0.5]:.3f}/{neg_dist['quantiles'][0.75]:.3f}")

    g2b = None
    if args.with_g2b or args.compare:
        g2b = load_g2b(args.g2b_data)
        print(f"나라장터 {len(g2b):,}건 ({g2b[F.DATE_COL].min():%Y-%m-%d} ~ {g2b[F.DATE_COL].max():%Y-%m-%d}) 가중치 사내 {W_INTERNAL:g}/시장 {W_G2B:g}")

    if args.compare:
        cv_a, _ = run_cv(df)
        cv_b, _ = run_cv(df, extra=g2b)
        rng = (f"{g2b[F.DATE_COL].min():%Y-%m-%d}", f"{g2b[F.DATE_COL].max():%Y-%m-%d}")
        print(write_compare(cv_a, cv_b, len(df), len(g2b), rng, args.compare_out))
        print(f"비교 리포트: {args.compare_out}")
        return 0

    extra = g2b if args.with_g2b else None
    cv, avg_iters = run_cv(df, extra=extra)

    # 최종 모델: 전체 데이터, CV 평균 반복수 (with-g2b 면 시장 행을 가중치로 포함)
    full = _prep_internal(df)
    if extra is not None:
        full = pd.concat([full, _prep_extra(extra, W_G2B)], ignore_index=True)
    enc = F.fit_encoders(full)
    feat = F.build_features(full, enc)
    X, y = feat[F.FEATURES], feat[F.TARGET]
    w = _norm_weights(feat[WEIGHT_COL], W_INTERNAL) if extra is not None else None
    models, _ = fit_quantile_models(X, y, n_estimators=avg_iters, w=w)
    imp = feature_importance(models)
    is_int = feat[F.SOURCE_COL] == "internal"
    feat_int, y_int = feat[is_int], y[is_int]           # 사전값·이력은 사내 행 기준
    train_mean = float(y_int.mean())

    text = write_report(cv, imp, len(df), avg_iters, args.report_out, n_extra=0 if extra is None else len(extra),
                        neg_dist=neg_dist)
    print(text)

    if not args.no_save:
        artifact = {
            "models": models,
            "features": F.FEATURES,
            "cat_features": F.CAT_FEATURES,
            "encoders": enc,
            "train_mean": train_mean,
            "priors": make_priors(y_int),
            "train_std": float(y_int.std(ddof=0)),
            "shrink_ml": SHRINK_ML,
            "n_estimators": avg_iters,
            "history_tail": feat_int[[F.DATE_COL, F.TARGET]].tail(F.ROLL_WINDOW).reset_index(drop=True),
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "source": "internal",
            "n_train": int(len(df)),
            "n_negotiated": int(len(neg)),
            "negotiated_dist": neg_dist,
            "with_g2b": extra is not None,
            "n_g2b": 0 if extra is None else int(len(extra)),
            "sample_weights": {"internal": W_INTERNAL, "g2b": W_G2B} if extra is not None else None,
            "data_range": [str(df[F.DATE_COL].min().date()), str(df[F.DATE_COL].max().date())],
            "cv_summary": cv.mean(numeric_only=True).to_dict(),
            "lgb_params": LGB_PARAMS,
        }
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.model_out)
        print(f"\n모델 저장: {args.model_out} | 리포트: {args.report_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
