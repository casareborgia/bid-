"""피처 생성의 미래 참조(누수) 방지 테스트."""
import numpy as np
import pandas as pd
import pytest

import features as F


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="7D")
    depts = rng.choice(["A본부", "B본부", "희귀"], size=n, p=[0.5, 0.45, 0.05])
    return pd.DataFrame({
        "contract_no": [f"C006{i:07d}" for i in range(n)],
        "contract_date": dates,
        "dept_name": depts,
        "work_type": rng.choice([1, 2], size=n),
        "award_method": rng.choice(["적격심사", "최저가"], size=n),
        "est_price": rng.uniform(1e7, 1e10, size=n),
        "bidder_count": rng.integers(1, 40, size=n).astype(float),
        "yega_rate": rng.normal(99.85, 0.6, size=n),
    })


def test_rolling_uses_only_past_rows():
    df = _df()
    enc = F.fit_encoders(df)
    feat = F.build_features(df, enc)
    r = feat["yega_rate"].values
    i = 30
    assert feat.loc[i, "ma_15_rate"] == pytest.approx(r[i - 15:i].mean())
    assert feat.loc[i, "std_15_rate"] == pytest.approx(r[i - 15:i].std(ddof=0))
    # 자기 자신은 포함되지 않는다: 행 i의 타깃을 바꿔도 행 i의 이동통계는 불변
    df2 = df.copy(); df2.loc[i, "yega_rate"] += 50
    feat2 = F.build_features(df2, enc)
    assert feat2.loc[i, "ma_15_rate"] == feat.loc[i, "ma_15_rate"]
    assert feat2.loc[i + 1, "ma_15_rate"] != feat.loc[i + 1, "ma_15_rate"]


def test_future_rows_do_not_affect_past_features():
    df = _df()
    enc = F.fit_encoders(df)
    base = F.build_features(df, enc)
    df2 = df.copy()
    df2.loc[df2.index >= 40, "yega_rate"] = 150.0          # 미래 구간만 오염
    df2.loc[df2.index >= 40, "bidder_count"] = 999
    after = F.build_features(df2, enc)
    cols = F.FEATURES
    pd.testing.assert_frame_equal(base.loc[:39, cols], after.loc[:39, cols])


def test_min_periods_and_order():
    df = _df().sample(frac=1, random_state=1)                # 섞어서 넣어도
    feat = F.build_features(df, F.fit_encoders(df))
    assert feat["contract_date"].is_monotonic_increasing      # 날짜순으로 정렬되고
    assert feat["ma_15_rate"].iloc[:F.ROLL_MIN_PERIODS].isna().all()   # 초기 구간은 NaN
    assert feat["ma_15_rate"].iloc[F.ROLL_MIN_PERIODS:].notna().all()


def test_encoders_merge_rare_and_unknown():
    df = _df()
    enc = F.fit_encoders(df)
    assert "희귀" not in enc["dept_code"] and F.OTHER in enc["dept_code"]
    feat = F.build_features(df, enc)
    assert set(feat.loc[df["dept_name"] == "희귀", "dept_code"].astype(str)) == {F.OTHER}
    row = F.make_inference_row(work_type=2, dept_name="없는본부", award_method="협상", est_price=1e9,
                               bidder_count=25, as_of=pd.Timestamp("2026-12-01"),
                               history_rates=feat["yega_rate"].tail(15), enc=enc)
    assert list(row.columns) == F.FEATURES
    assert str(row["dept_code"].iloc[0]) == F.OTHER and str(row["award_method"].iloc[0]) == F.OTHER
    assert row["is_year_end"].iloc[0] == 1 and row["est_price_log"].iloc[0] == pytest.approx(9.0)
    assert row["ma_15_rate"].iloc[0] == pytest.approx(feat["yega_rate"].tail(15).mean())


def test_forbidden_features_absent():
    for bad in ("branch_code", "base_design_ratio", "bid_rate_yega", "yega_amt", "base_amt"):
        assert bad not in F.FEATURES
    assert 6 <= len(F.FEATURES) <= 8


def test_rolling_is_computed_per_source():
    """source 컬럼이 있으면 사내 행의 이동통계는 사내 이력만, g2b 행은 g2b 이력만 본다."""
    a = _df(n=40, seed=1); a["source"] = "internal"
    b = _df(n=40, seed=2); b["source"] = "g2b"
    b["contract_no"] = "G" + b["contract_no"]
    b["contract_date"] = b["contract_date"] + pd.Timedelta(days=3)      # 날짜가 서로 끼어들도록
    both = pd.concat([a, b], ignore_index=True)
    enc = F.fit_encoders(both)
    feat = F.build_features(both, enc)
    only_a = F.build_features(a, enc).set_index("contract_no")
    only_b = F.build_features(b, enc).set_index("contract_no")
    fa = feat[feat["source"] == "internal"].set_index("contract_no")
    fb = feat[feat["source"] == "g2b"].set_index("contract_no")
    pd.testing.assert_series_equal(fa["ma_15_rate"], only_a.loc[fa.index, "ma_15_rate"], check_names=False)
    pd.testing.assert_series_equal(fb["std_15_rate"], only_b.loc[fb.index, "std_15_rate"], check_names=False)
    # 단일 source 면 source 컬럼이 없을 때와 완전히 같다
    pd.testing.assert_frame_equal(F.build_features(a, enc)[F.FEATURES], F.build_features(a.drop(columns="source"), enc)[F.FEATURES])
