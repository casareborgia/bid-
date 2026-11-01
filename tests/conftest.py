import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------- Phase 7 공용 픽스처 (합성 데이터 소형 아티팩트)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def _internal_df(n=400, seed=7):
    rng = np.random.default_rng(seed)
    bidders = rng.integers(1, 60, size=n).astype(float)
    return pd.DataFrame({
        "contract_no": [f"C006{i:07d}" for i in range(n)],
        "contract_date": pd.date_range("2021-01-01", periods=n, freq="3D"),
        "dept_name": rng.choice(["A본부", "B본부"], size=n),
        "work_type": rng.choice([1, 2], size=n),
        "award_method": rng.choice(["적격심사", "최저가"], size=n),
        "est_price": rng.uniform(1e7, 1e10, size=n),
        "bidder_count": bidders,
        "yega_rate": 99.85 - 0.004 * bidders + rng.normal(0, 0.5, size=n),
    })


def _market_df(n=1200, seed=11, start="2024-01-01", freq="4h"):
    rng = np.random.default_rng(seed)
    rc = rng.choice(["±2", "±3"], size=n, p=[0.3, 0.7])
    sd = np.where(rc == "±2", 0.5, 0.74)
    return pd.DataFrame({
        "contract_no": [f"R{i:06d}|000|0|000" for i in range(n)],
        "contract_date": pd.date_range(start, periods=n, freq=freq),
        "dept_name": "시장", "award_method": "기타",
        "work_type": rng.choice([1, 2], size=n), "range_cls": rc,
        "est_price": rng.uniform(1e7, 1e10, size=n),
        "bidder_count": rng.integers(2, 120, size=n).astype(float),
        "yega_rate": np.where(rc == "±2", 100.0, 99.85) + rng.normal(0, 1, size=n) * sd,
    })


@pytest.fixture(scope="session")
def synthetic_internal_df():
    return _internal_df()


@pytest.fixture(scope="session")
def synthetic_market_df():
    return _market_df()


@pytest.fixture(scope="session")
def internal_model_path(tmp_path_factory, synthetic_internal_df):
    import joblib
    import features as F
    import train_model as T
    df = synthetic_internal_df
    enc = F.fit_encoders(df); feat = F.build_features(df, enc)
    X, y = feat[F.FEATURES], feat[F.TARGET]
    models, _ = T.fit_quantile_models(X, y, n_estimators={a: 30 for a in T.QUANTILES})
    neg = pd.DataFrame({"yega_rate": np.random.default_rng(1).normal(98.5, 1.2, size=40), "work_type": 2,
                        "contract_date": pd.date_range("2021-01-01", periods=40, freq="30D")})
    art = {"source": "internal", "models": models, "features": F.FEATURES, "cat_features": F.CAT_FEATURES, "encoders": enc,
           "train_mean": float(y.mean()), "train_std": float(y.std(ddof=0)), "priors": T.make_priors(y), "shrink_ml": T.SHRINK_ML,
           "history_tail": feat[[F.DATE_COL, F.TARGET]].tail(F.ROLL_WINDOW).reset_index(drop=True),
           "trained_at": "test", "n_train": len(df), "data_range": ["2021-01-01", "2024-04-13"],
           "n_negotiated": 40, "negotiated_dist": T.negotiated_distribution(neg),
           "cv_summary": {"baseline_mae_val": 0.4, "ml_mae_val": 0.4, "gap": 0.0, "coverage_q25_q75": 0.5}}
    p = tmp_path_factory.mktemp("art") / "bidding_model.joblib"; joblib.dump(art, p); return p


@pytest.fixture(scope="session")
def market_model_path(tmp_path_factory, synthetic_market_df):
    import joblib
    import features as F
    import train_model as T
    df = synthetic_market_df
    enc = F.fit_encoders(df); feat = F.build_features(df, enc)
    X, y = feat[F.MARKET_FEATURES], feat[F.TARGET]
    models, _ = T.fit_quantile_models(X, y, n_estimators={a: 30 for a in T.QUANTILES}, cat_features=F.MARKET_CAT_FEATURES)
    gp = T.make_group_priors(y, feat["range_cls"].astype(str))
    art = {"source": "market", "models": models, "features": F.MARKET_FEATURES, "cat_features": F.MARKET_CAT_FEATURES,
           "encoders": enc, "range_classes": F.RANGE_CLASSES, "priors_by_range": gp, "priors": gp["±3"],
           "train_mean": float(y.mean()), "train_std": float(y.std(ddof=0)), "bidder_median": 30.0,
           "shrink_ml": T.SHRINK_ML, "history_tail": None, "trained_at": "test", "n_train": len(df),
           "n_negotiated": 0, "negotiated_dist": None, "data_range": ["2024-01-01", "2024-07-19"],
           "cv_summary": {"baseline_mae_val": 0.57, "ml_mae_val": 0.57, "gap": 0.0, "coverage_q25_q75": 0.5}}
    p = tmp_path_factory.mktemp("art") / "market_model.joblib"; joblib.dump(art, p); return p
