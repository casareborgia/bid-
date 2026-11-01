"""학습 모듈 스모크 테스트 (합성 데이터, 네트워크·실데이터 불필요)."""
import numpy as np
import pandas as pd

import features as F
import train_model as T


def _df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="3D")
    bidders = rng.integers(1, 60, size=n).astype(float)
    rate = 99.85 - 0.004 * bidders + rng.normal(0, 0.5, size=n)   # 업체수 많을수록 약간 낮게
    return pd.DataFrame({
        "contract_no": [f"C006{i:07d}" for i in range(n)],
        "contract_date": dates,
        "dept_name": rng.choice(["A본부", "B본부"], size=n),
        "work_type": rng.choice([1, 2], size=n),
        "award_method": rng.choice(["적격심사", "최저가"], size=n),
        "est_price": rng.uniform(1e7, 1e10, size=n),
        "bidder_count": bidders,
        "yega_rate": rate,
    })


def test_params_respect_overfit_limits():
    p = T.LGB_PARAMS
    assert p["max_depth"] <= 3 and p["num_leaves"] <= 7 and p["min_child_samples"] >= 30
    assert p["subsample"] == 0.8 and p["n_estimators"] <= 300
    assert T.SHRINK_ML == 0.7


def test_fit_predict_quantiles_ordered_and_shrunk():
    df = _df()
    enc = F.fit_encoders(df)
    feat = F.build_features(df, enc).dropna(subset=["ma_15_rate"])
    X, y = feat[F.FEATURES], feat[F.TARGET]
    models, best = T.fit_quantile_models(X, y)
    assert set(models) == set(T.QUANTILES) and all(b >= 1 for b in best.values())
    priors = T.make_priors(y)
    assert priors[0.25] < priors[0.5] < priors[0.75]
    p = T.predict_quantiles(models, X, priors)
    assert (p[0.25] <= p[0.5]).all() and (p[0.5] <= p[0.75]).all()
    # 축소: 원시 예측과 사전값 사이에 있어야 한다
    raw = models[0.5].predict(X)
    assert np.allclose(p[0.5], 0.7 * raw + 0.3 * priors[0.5]) or (np.abs(p[0.5] - (0.7 * raw + 0.3 * priors[0.5])) < 1e-9).all()
    # 재현성: 반복수를 고정하면 같은 결과
    models2, _ = T.fit_quantile_models(X, y, n_estimators=best)
    assert np.allclose(models2[0.5].predict(X), raw)


def test_run_cv_outputs_required_columns():
    cv, iters = T.run_cv(_df(n=600))
    assert len(cv) == T.N_SPLITS
    for c in ("baseline_mae_val", "ml_mae_val", "gap", "improve_pct", "coverage_q25_q75", "coverage_raw",
              "pass_q25", "pass_q50", "pass_q75"):
        assert c in cv.columns and cv[c].notna().all()
    # 기준별 통과율 = P(y ≤ Qα): 단조 증가하고 커버리지와 정합 (pass_q75 - pass_q25 == Q25~Q75 커버리지, 경계 포함 차이 허용)
    assert (cv["pass_q25"] <= cv["pass_q50"]).all() and (cv["pass_q50"] <= cv["pass_q75"]).all()
    assert np.allclose(cv["pass_q75"] - cv["pass_q25"], cv["coverage_q25_q75"], atol=0.01)
    assert cv["val_n"].sum() + cv["train_n"].iloc[0] == 600
    assert all(v >= 20 for v in iters.values())


def _g2b(n=300, seed=5, start="2021-06-01"):
    """collect_g2b_api.build() 가 만드는 parquet 과 같은 컬럼의 합성 시장 데이터."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(1e7, 5e9, size=n)
    rate = rng.normal(99.85, 0.6, size=n)
    rate[:3] = [90.0, 105.0, 99.0]                        # 필터 확인용 이상치 2건
    return pd.DataFrame({
        "bid_key": [f"R{i:06d}|000|0|000" for i in range(n)],
        "work_type": rng.choice([1, 2], size=n),
        "base_amt": base,
        "yega_amt": base * rate / 100,
        "yega_rate": rate,
        "open_dt": pd.date_range(start, periods=n, freq="4D"),
        "bidder_count": rng.integers(1, 30, size=n).astype(float),
        "source": "g2b",
    })


def test_g2b_to_internal_schema_filters_and_maps():
    g = _g2b()
    out = T.g2b_to_internal_schema(g)
    assert len(out) == len(g) - 2                                 # 96.5~103.5 밖 2건 제외
    for c in ("contract_no", "contract_date", "dept_name", "work_type", "award_method", "est_price", "bidder_count", "yega_rate", "source"):
        assert c in out.columns
    assert (out["dept_name"] == T.G2B_DEPT).all() and (out["award_method"] == T.G2B_AWARD).all()
    assert (out["source"] == "g2b").all()
    kept = g[g["yega_rate"].between(T.RATE_LO, T.RATE_HI)]
    assert np.allclose(out["est_price"], kept["base_amt"].values / T.EST_PRICE_DIV)
    assert (out["contract_date"].values == kept["open_dt"].values).all()
    assert "bid_rate_yega" not in out.columns and "base_amt" not in out.columns


def test_run_cv_with_extra_evaluates_internal_only_and_respects_cutoff():
    df, g = _df(n=600), T.g2b_to_internal_schema(_g2b(n=300))
    cv_a, _ = T.run_cv(df)
    cv_b, _ = T.run_cv(df, extra=g)
    # 폴드·검증 집합·Baseline 은 사내 행 기준이라 동일
    assert cv_a["val_n"].tolist() == cv_b["val_n"].tolist() and cv_a["train_n"].tolist() == cv_b["train_n"].tolist()
    assert np.allclose(cv_a["baseline_mae_val"], cv_b["baseline_mae_val"])
    assert (cv_a["extra_n"] == 0).all()
    # 폴드의 g2b 학습행은 사내 학습구간 마지막 날짜 이전 것만 → 폴드가 커질수록 단조 증가
    assert cv_b["extra_n"].is_monotonic_increasing and cv_b["extra_n"].iloc[-1] > 0
    df_sorted = df.sort_values("contract_date")
    n_int = len(df)
    fold1_train_max = df_sorted["contract_date"].iloc[cv_b["train_n"].iloc[0] - 1]
    assert cv_b["extra_n"].iloc[0] == int((g["contract_date"] <= fold1_train_max).sum())
    assert cv_b["val_n"].sum() + cv_b["train_n"].iloc[0] == n_int
    # 시장 데이터가 섞이면 예측이 달라져야 한다(가중치가 실제로 적용됨)
    assert not np.allclose(cv_a["ml_mae_val"], cv_b["ml_mae_val"])


def test_fit_with_sample_weight_changes_model():
    df = _df(n=400)
    feat = F.build_features(df, F.fit_encoders(df)).dropna(subset=["ma_15_rate"])
    X, y = feat[F.FEATURES], feat[F.TARGET]
    w = pd.Series(np.where(np.arange(len(X)) % 2 == 0, 3.0, 0.1), index=X.index)
    m1, _ = T.fit_quantile_models(X, y, n_estimators={a: 50 for a in T.QUANTILES})
    m2, _ = T.fit_quantile_models(X, y, n_estimators={a: 50 for a in T.QUANTILES}, w=w)
    assert not np.allclose(m1[0.5].predict(X), m2[0.5].predict(X))


def test_split_negotiated_and_distribution():
    df = _df(n=200)
    # contract_method 가 없으면 전부 경쟁입찰
    comp, neg = T.split_negotiated(df)
    assert len(comp) == 200 and neg.empty and T.negotiated_distribution(neg) is None
    # 수의 행은 분리되고, 경험 분포가 만들어진다
    df["contract_method"] = "제한경쟁"
    df.loc[df.index[:20], "contract_method"] = "수의"
    df.loc[df.index[:20], "yega_rate"] = 98.0
    df.loc[df.index[20:23], "contract_method"] = " 수의계약 "          # 공백·접미어 허용
    comp, neg = T.split_negotiated(df)
    assert len(neg) == 23 and len(comp) == 177
    assert not comp["contract_method"].str.contains("수의").any()
    dist = T.negotiated_distribution(neg)
    assert dist["n"] == 23 and set(dist["quantiles"]) == set(T.NEG_QUANTILES)
    assert dist["quantiles"][0.25] <= dist["quantiles"][0.5] <= dist["quantiles"][0.75]
    assert abs(dist["quantiles"][0.5] - 98.0) < 0.5                       # 수의 행 대부분이 98.0
    assert set(dist["by_work_type"]) <= {1, 2} and "date_range" in dist
    # 경쟁입찰만 학습하면 수의 행의 낮은 사정률이 사전값에 섞이지 않는다
    assert T.make_priors(comp["yega_rate"])[0.5] > T.make_priors(df["yega_rate"])[0.5]


def test_write_report_includes_negotiated_section(tmp_path):
    df = _df(n=600)
    cv, iters = T.run_cv(df)
    feat = F.build_features(df, F.fit_encoders(df))
    models, _ = T.fit_quantile_models(feat[F.FEATURES], feat[F.TARGET], n_estimators=iters)
    imp = T.feature_importance(models)
    neg = df.head(30).copy(); neg["contract_method"] = "수의"
    text = T.write_report(cv, imp, len(df), iters, tmp_path / "r.md", neg_dist=T.negotiated_distribution(neg))
    assert "경쟁입찰 600건" in text and "수의계약 30건은 학습 제외" in text and "## 수의계약" in text
    text2 = T.write_report(cv, imp, len(df), iters, tmp_path / "r2.md")
    assert "## 수의계약" not in text2


def _market_raw(n=600, seed=9):
    """collect_g2b_api.build() parquet 과 같은 컬럼(range_cls 포함)의 합성 시장 데이터."""
    rng = np.random.default_rng(seed)
    rc = rng.choice(["±2", "±3", "±1", None], size=n, p=[0.3, 0.66, 0.02, 0.02])
    sd = np.where(rc == "±2", 0.5, 0.74)
    base = rng.uniform(1e7, 5e9, size=n)
    rate = np.where(rc == "±2", 100.0, 99.85) + rng.normal(0, 1, size=n) * sd
    bidders = rng.integers(2, 120, size=n).astype(float); bidders[:50] = np.nan
    return pd.DataFrame({
        "bid_key": [f"R{i:06d}|000|0|000" for i in range(n)], "work_type": rng.choice([1, 2], size=n),
        "base_amt": base, "yega_amt": base * rate / 100, "yega_rate": rate, "range_cls": rc,
        "open_dt": pd.date_range("2024-01-01", periods=n, freq="6h"), "bidder_count": bidders, "source": "g2b",
    })


def test_market_schema_filters_and_fills():
    g = _market_raw()
    df, meta = T.market_schema(g)
    assert set(df["range_cls"]) == {"±2", "±3"} and meta["n_used"] == len(df) < len(g)
    assert df["bidder_count"].notna().all() and meta["bidder_median"] > 1 and 0 < meta["bidder_missing_frac"] < 0.2
    assert (df["bidder_count"] == meta["bidder_median"]).sum() >= 1
    assert df["contract_date"].is_monotonic_increasing and (df["dept_name"] == T.G2B_DEPT).all()
    rs = T.range_stats(df)
    assert list(rs.index) == ["±2", "±3"] and rs.loc["±3", "std"] > rs.loc["±2", "std"]


def test_group_priors_and_market_cv():
    df, _ = T.market_schema(_market_raw(n=900))
    gp = T.make_group_priors(df["yega_rate"], df["range_cls"])
    assert set(gp) == {"±2", "±3"} and gp["±2"][0.5] > gp["±3"][0.5]
    rows = T.priors_for_rows(gp, df["range_cls"].head(5))
    assert all(len(v) == 5 for v in rows.values())
    assert rows[0.5][0] == gp[str(df["range_cls"].iloc[0])][0.5]
    cv, iters = T.run_cv(df, features=F.MARKET_FEATURES, cat_features=F.MARKET_CAT_FEATURES, prior_group_col=F.RANGE_COL)
    assert len(cv) == T.N_SPLITS and cv["ml_mae_val"].notna().all() and cv["coverage_q25_q75"].between(0.3, 0.7).all()
    # 그룹별 Baseline 은 단일 평균 Baseline 보다 낫거나 같아야 한다(±2/±3 평균이 다르므로)
    cv1, _ = T.run_cv(df, features=F.MARKET_FEATURES, cat_features=F.MARKET_CAT_FEATURES)
    assert cv["baseline_mae_val"].mean() <= cv1["baseline_mae_val"].mean() + 1e-9


def test_write_compare_reports_verdict(tmp_path):
    df, g = _df(n=600), T.g2b_to_internal_schema(_g2b(n=300))
    cv_a, _ = T.run_cv(df)
    cv_b, _ = T.run_cv(df, extra=g)
    text = T.write_compare(cv_a, cv_b, len(df), len(g), ("2021-06-01", "2024-09-01"), tmp_path / "cv_compare.md")
    assert (tmp_path / "cv_compare.md").exists()
    assert "3a ML MAE" in text and "3b ML MAE" in text and "## 결론" in text
    assert ("3b 채택" in text) != ("3a 유지" in text)
    adopt, reason = T.compare_verdict(cv_a, cv_b)
    assert ("3b 채택" in text) == adopt and "검증 MAE 평균" in reason
