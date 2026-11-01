"""Phase 4 CLI 테스트. 합성 데이터로 만든 소형 모델로 계산 규칙을 검증하고, 실제 모델이 있으면 스모크 실행."""
import json

import numpy as np
import pandas as pd
import pytest

import features as F
import predict_cli as P
import train_model as T


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    rng = np.random.default_rng(7)
    n = 400
    bidders = rng.integers(1, 60, size=n).astype(float)
    df = pd.DataFrame({
        "contract_no": [f"C006{i:07d}" for i in range(n)],
        "contract_date": pd.date_range("2021-01-01", periods=n, freq="3D"),
        "dept_name": rng.choice(["A본부", "B본부"], size=n),
        "work_type": rng.choice([1, 2], size=n),
        "award_method": rng.choice(["적격심사", "최저가"], size=n),
        "est_price": rng.uniform(1e7, 1e10, size=n),
        "bidder_count": bidders,
        "yega_rate": 99.85 - 0.004 * bidders + rng.normal(0, 0.5, size=n),
    })
    enc = F.fit_encoders(df)
    feat = F.build_features(df, enc)
    X, y = feat[F.FEATURES], feat[F.TARGET]
    models, best = T.fit_quantile_models(X, y)
    neg = pd.DataFrame({"yega_rate": rng.normal(98.5, 1.2, size=40), "work_type": rng.choice([1, 2], size=40),
                        "contract_date": pd.date_range("2021-01-01", periods=40, freq="30D")})
    art = {"models": models, "features": F.FEATURES, "cat_features": F.CAT_FEATURES, "encoders": enc,
           "train_mean": float(y.mean()), "priors": T.make_priors(y), "shrink_ml": T.SHRINK_ML,
           "history_tail": feat[[F.DATE_COL, F.TARGET]].tail(F.ROLL_WINDOW).reset_index(drop=True),
           "trained_at": "test", "n_train": n, "data_range": ["2021-01-01", "2024-04-13"],
           "n_negotiated": 40, "negotiated_dist": T.negotiated_distribution(neg)}
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    import joblib; joblib.dump(art, path)
    return path


def _args(**kw):
    import argparse
    d = dict(est_price=1_000_000_000, work_type=2, bidder_count=25, lower_limit=87.745, award_method="적격심사",
             dept=None, as_of="2024-06-01", margin=0.1, round_unit=1000, negotiated=False, basis="Q75", range=None)
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture(scope="module")
def market_model(tmp_path_factory):
    """train_model.train_market 와 같은 구조의 소형 시장 아티팩트 (합성 데이터)."""
    rng = np.random.default_rng(11)
    n = 1200
    rc = rng.choice(["±2", "±3"], size=n, p=[0.3, 0.7])
    sd = np.where(rc == "±2", 0.5, 0.74)
    df = pd.DataFrame({
        "contract_no": [f"R{i:06d}|000|0|000" for i in range(n)],
        "contract_date": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "dept_name": "시장", "award_method": "기타",
        "work_type": rng.choice([1, 2], size=n), "range_cls": rc,
        "est_price": rng.uniform(1e7, 1e10, size=n),
        "bidder_count": rng.integers(2, 120, size=n).astype(float),
        "yega_rate": np.where(rc == "±2", 100.0, 99.85) + rng.normal(0, 1, size=n) * sd,
    })
    enc = F.fit_encoders(df)
    feat = F.build_features(df, enc)
    X, y = feat[F.MARKET_FEATURES], feat[F.TARGET]
    models, _ = T.fit_quantile_models(X, y, n_estimators={a: 30 for a in T.QUANTILES}, cat_features=F.MARKET_CAT_FEATURES)
    gp = T.make_group_priors(y, feat["range_cls"].astype(str))
    art = {"source": "market", "models": models, "features": F.MARKET_FEATURES, "cat_features": F.MARKET_CAT_FEATURES,
           "encoders": enc, "range_classes": F.RANGE_CLASSES, "priors_by_range": gp, "priors": gp["±3"],
           "train_mean": float(y.mean()), "train_std": float(y.std(ddof=0)), "bidder_median": 30.0,
           "shrink_ml": T.SHRINK_ML, "history_tail": None, "trained_at": "test", "n_train": n,
           "n_negotiated": 0, "negotiated_dist": None, "data_range": ["2024-01-01", "2024-07-19"]}
    path = tmp_path_factory.mktemp("mk") / "market_model.joblib"
    import joblib; joblib.dump(art, path)
    return path


def test_market_mode_uses_range_specific_priors(market_model):
    art = P.load_model(market_model)
    r3 = P.build_report(art, _args(range=3, bidder_count=None))
    r2 = P.build_report(art, _args(range=2, bidder_count=None))
    assert r3["model"]["mode"] == "market" and r3["input"]["market"] is True and r3["input"]["range_cls"] == "±3"
    assert r3["input"]["bidder_count"] == 30.0                      # 참여업체수 생략 → 시장 중앙값
    q3 = {x["quantile"]: x["yega_rate"] for x in r3["quantiles"]}
    q2 = {x["quantile"]: x["yega_rate"] for x in r2["quantiles"]}
    assert q3["Q75"] - q3["Q25"] > q2["Q75"] - q2["Q25"]              # ±3% 구간이 ±2% 보다 넓다
    assert q2["Q50"] > q3["Q50"]                                      # ±2% 평균 100.0 > ±3% 평균 99.85
    assert r3["model"]["dept_mapped"] is None and r3["model"]["history_n"] == 0
    with pytest.raises(ValueError):                                   # 시장 모델은 --range 필수
        P.build_report(art, _args(range=None))
    with pytest.raises(ValueError):                                   # 시장 모델에 --negotiated 불가
        P.build_report(art, _args(range=3, negotiated=True))


def test_market_cli_flags(small_model, market_model, capsys):
    rc = P.main(["--est_price", "1000000000", "--market", "--range", "3", "--model", str(market_model), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["mode"] == "market" and out["recommended"]["basis"] == "Q75"
    with pytest.raises(SystemExit):                                   # --market 인데 --range 없음
        P.main(["--est_price", "100", "--market", "--model", str(market_model)])
    with pytest.raises(SystemExit):                                   # 사내 모델에 --range
        P.main(["--est_price", "100", "--range", "3", "--model", str(small_model)])
    rc = P.main(["--est_price", "100", "--market", "--range", "3", "--model", str(small_model)])
    assert rc == 1                                                    # 시장 모델이 아닌 파일


def test_ceil_to():
    assert P.ceil_to(1234567, 1000) == 1235000
    assert P.ceil_to(1235000, 1000) == 1235000
    assert P.ceil_to(12.3, 1) == 13


def test_report_rules(small_model):
    art = P.load_model(small_model)
    r = P.build_report(art, _args())
    assert r["base_amt"] == 1_100_000_000                       # 추정가격 × 1.1
    q = {x["quantile"]: x for x in r["quantiles"]}
    assert q["Q25"]["yega_rate"] <= q["Q50"]["yega_rate"] <= q["Q75"]["yega_rate"]
    assert 96.5 < q["Q25"]["yega_rate"] and q["Q75"]["yega_rate"] < 103.5
    for x in r["quantiles"]:
        assert x["yega_amt"] == int(round(r["base_amt"] * x["yega_rate"] / 100))
        assert x["lower_limit_amt"] == int(round(x["yega_amt"] * 87.745 / 100))
        assert x["bid_amt"] >= x["lower_limit_amt"] * 1.001 - 1 and x["bid_amt"] % 1000 == 0
        assert x["bid_amt"] - x["lower_limit_amt"] * 1.001 < 1000
    # 권장 기준은 Q75(안전). Qα 기준 투찰가의 통과확률은 α 다 (실제 사정률 ≤ Qα 일 때만 하한을 넘는다)
    assert r["recommended"]["basis"] == "Q75" and r["recommended"]["bid_amt"] == q["Q75"]["bid_amt"]
    assert r["recommended"]["pass_prob"] == 0.75
    assert q["Q25"]["pass_prob"] == 0.25 and q["Q50"]["pass_prob"] == 0.5 and q["Q75"]["pass_prob"] == 0.75
    assert q["Q25"]["bid_amt"] <= q["Q50"]["bid_amt"] <= q["Q75"]["bid_amt"]
    assert r["model"]["dept_mapped"] == F.OTHER                 # 부서 미지정 → 기타
    assert r["model"]["history_n"] == F.ROLL_WINDOW
    assert r["model"]["mode"] == "ml" and r["input"]["negotiated"] is False


def test_negotiated_mode_uses_empirical_distribution(small_model):
    art = P.load_model(small_model)
    r = P.build_report(art, _args(negotiated=True))
    dist = art["negotiated_dist"]["quantiles"]
    q = {x["quantile"]: x for x in r["quantiles"]}
    assert r["model"]["mode"] == "negotiated" and r["input"]["negotiated"] is True
    assert q["Q25"]["yega_rate"] == round(dist[0.25], 4) and q["Q50"]["yega_rate"] == round(dist[0.5], 4)
    assert q["Q25"]["yega_rate"] <= q["Q50"]["yega_rate"] <= q["Q75"]["yega_rate"]
    assert r["model"]["negotiated_dist"]["n"] == 40 and r["model"]["raw_rates"] is None
    # 금액 계산 규칙은 ML 모드와 동일
    for x in r["quantiles"]:
        assert x["yega_amt"] == int(round(r["base_amt"] * x["yega_rate"] / 100))
        assert x["bid_amt"] % 1000 == 0
    # 분포가 없는 아티팩트면 명확한 오류
    art2 = {**art, "negotiated_dist": None}
    with pytest.raises(ValueError):
        P.build_report(art2, _args(negotiated=True))


def test_pass_probability_direction(small_model):
    """Q25 기준 투찰가는 실제 사정률이 중앙값(Q50)으로만 나와도 낙찰하한에 미달한다. 즉 통과확률은 75%가 아니라 25%다."""
    art = P.load_model(small_model)
    r = P.build_report(art, _args(margin=0.0))
    q = {x["quantile"]: x for x in r["quantiles"]}
    actual_lower_at_q50 = q["Q50"]["lower_limit_amt"]              # 실제 예정가격이 Q50 이면 이것이 실제 하한금액
    assert q["Q25"]["bid_amt"] < actual_lower_at_q50                # Q25 기준 투찰가 → 하한 미달
    assert q["Q75"]["bid_amt"] >= actual_lower_at_q50               # Q75 기준 투찰가 → 통과
    # --basis 로 기준 변경
    r2 = P.build_report(art, _args(basis="Q25"))
    assert r2["recommended"]["basis"] == "Q25" and r2["recommended"]["pass_prob"] == 0.25
    assert r2["recommended"]["bid_amt"] == {x["quantile"]: x for x in r2["quantiles"]}["Q25"]["bid_amt"]


def test_as_of_filters_history_and_unknown_categories(small_model):
    art = P.load_model(small_model)
    r = P.build_report(art, _args(as_of="2020-01-01", dept="없는부서", award_method="듣보"))
    assert r["model"]["history_n"] == 0                          # 기준일 이전 이력 없음 → NaN 피처로 예측
    assert r["model"]["award_mapped"] == F.OTHER
    assert all(96 < x["yega_rate"] < 104 for x in r["quantiles"])


def test_cli_json_and_validation(small_model, capsys):
    rc = P.main(["--est_price", "1000000000", "--model", str(small_model), "--json", "--as_of", "2024-06-01"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["base_amt"] == 1_100_000_000 and len(out["quantiles"]) == 3
    rc = P.main(["--est_price", "1000000000", "--model", str(small_model), "--json", "--negotiated"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["model"]["mode"] == "negotiated"
    rc = P.main(["--est_price", "1000000000", "--model", str(small_model), "--json", "--basis", "Q50"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["recommended"]["basis"] == "Q50"
    with pytest.raises(SystemExit):
        P.main(["--est_price", "-5", "--model", str(small_model)])
    with pytest.raises(SystemExit):
        P.main(["--est_price", "100", "--lower_limit", "150", "--model", str(small_model)])


@pytest.mark.skipif(not P.MODEL_PATH.exists(), reason="실제 모델 없음")
def test_real_model_smoke(capsys):
    rc = P.main(["--est_price", "1000000000", "--work_type", "2", "--bidder_count", "25", "--lower_limit", "87.745", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    q = {x["quantile"]: x["yega_rate"] for x in out["quantiles"]}
    assert 99.0 < q["Q25"] < q["Q50"] < q["Q75"] < 101.0


def test_lower_limit_amount_with_a_value():
    """A값은 하한율을 곱하지 않는다: (예정가격 - A) × 하한율 + A. A=0 이면 기존 식과 같다."""
    assert P.lower_limit_amount(550_000_000, 89.745, 30_000_000) == pytest.approx(496_674_000)
    assert P.lower_limit_amount(550_000_000, 89.745, 0) == pytest.approx(550_000_000 * 0.89745)


def test_build_report_a_value(small_model):
    import joblib
    art = joblib.load(small_model)
    base = P.build_report(art, _args(work_type=1, lower_limit=89.745))
    with_a = P.build_report(art, _args(work_type=1, lower_limit=89.745, a_value=30_000_000))
    assert with_a["input"]["a_value"] == 30_000_000
    for q0, qa in zip(base["quantiles"], with_a["quantiles"]):
        assert q0["yega_amt"] == qa["yega_amt"] and q0["pass_prob"] == qa["pass_prob"]  # 예측·통과확률은 그대로
        assert qa["lower_limit_amt"] == round(P.lower_limit_amount(qa["yega_amt"], 89.745, 30_000_000))
        assert qa["bid_amt"] > q0["bid_amt"] and qa["bid_amt"] % 1000 == 0              # 하한가·투찰가는 올라간다
        assert qa["bid_amt"] >= qa["lower_limit_amt"] * 1.001
    with pytest.raises(ValueError, match="공사"):
        P.build_report(art, _args(work_type=2, a_value=30_000_000))
    with pytest.raises(ValueError, match="0 이상"):
        P.build_report(art, _args(work_type=1, a_value=-1))
    with pytest.raises(ValueError, match="예정가격 추정치"):
        P.build_report(art, _args(work_type=1, a_value=10_000_000_000))


def test_cli_a_value_rejected_for_service(small_model):
    with pytest.raises(SystemExit):
        P.main(["--est_price", "1000000000", "--work_type", "2", "--a_value", "30000000", "--model", str(small_model)])


NOTICE_CNSTWK = {
    "bid_no": "R26BK01752065", "bid_ord": "000", "name": "마곡나루역 펀스테이션 전기공사", "work_type": 1,
    "notice_date": "2026-10-01", "agency": "서울특별시", "est_price": 167_746_809.0, "base_amt": 184_521_490.0,
    "range_pct": 3, "lower_limit_pct": 89.745, "a_value": 11_684_881.0, "a_items": {"국민연금보험료": 3_583_072.0},
    "servc_kind": None, "award_method_name": "적격심사", "contract_method": "제한경쟁",
    "open_dt": "2026-10-13 13:00:00", "warnings": [],
}


def _notice_args(notice=NOTICE_CNSTWK, **kw):
    import argparse
    args = argparse.Namespace(est_price=None, work_type=None, bidder_count=None, lower_limit=None, award_method="적격심사",
                              dept=None, agency=None, as_of=None, margin=0.1, round_unit=1000, negotiated=False,
                              basis="Q75", range=None, a_value=None, servc_kind=None)
    for k, v in kw.items():
        setattr(args, k, v)
    P.apply_notice(args, notice)
    return args


def test_build_report_uses_notice_values(market_model):
    import joblib
    art = joblib.load(market_model)
    r = P.build_report(art, _notice_args())
    i = r["input"]
    assert r["base_amt"] == 184_521_490 and i["base_amt_source"] == "notice"       # ×1.1 대신 공고 기초금액
    assert i["range_cls"] == "±3" and i["range_source"] == "notice"
    assert i["lower_limit_pct"] == 89.745 and i["lower_limit_source"] == "notice"
    assert i["a_value"] == 11_684_881 and i["a_source"] == "notice"
    assert i["work_type"] == 1 and i["agency"] == "서울특별시" and i["as_of"] == "2026-10-01"
    assert r["notice"]["bid_no"] == "R26BK01752065"
    q = r["quantiles"][2]
    assert q["lower_limit_amt"] == round(P.lower_limit_amount(q["yega_amt"], 89.745, 11_684_881))


def test_explicit_args_override_notice(market_model):
    import joblib
    art = joblib.load(market_model)
    r = P.build_report(art, _notice_args(range=2, lower_limit=88.0, a_value=0.0, est_price=100_000_000.0))
    i = r["input"]
    assert i["range_source"] == "explicit" and i["range_cls"] == "±2"
    assert i["lower_limit_source"] == "explicit" and i["lower_limit_pct"] == 88.0
    assert i["a_value"] == 0 and i["a_source"] == "explicit"
    assert i["est_price"] == 100_000_000


def test_notice_lower_limit_differs_from_rules_is_flagged(market_model):
    """공고 하한율(예: 소액수의 88%)이 규칙표 판정과 다르면 공고값을 쓰고 근거에 차이를 남긴다."""
    import joblib
    art = joblib.load(market_model)
    n = dict(NOTICE_CNSTWK, lower_limit_pct=88.0)
    r = P.build_report(art, _notice_args(n))
    assert r["input"]["lower_limit_pct"] == 88.0
    assert "규칙표 판정" in r["input"]["lower_limit_basis"] and "공고값 우선" in r["input"]["lower_limit_basis"]


def test_notice_without_base_amount_falls_back(market_model):
    import joblib
    art = joblib.load(market_model)
    n = dict(NOTICE_CNSTWK, base_amt=None, range_pct=None, a_value=0.0)
    r = P.build_report(art, _notice_args(n))
    assert r["base_amt"] == round(167_746_809 * 1.1) and r["input"]["base_amt_source"] == "est_price×1.1"
    assert r["input"]["range_source"] == "auto"                                     # 서울특별시 → 규칙표 판정


def test_cli_bid_no(market_model, monkeypatch, capsys):
    import bid_notice as BN
    monkeypatch.setattr(BN, "fetch_notice", lambda bid_no, client=None: dict(NOTICE_CNSTWK))
    assert P.main(["--bid_no", "R26BK01752065", "--model", str(market_model), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["mode"] == "market" and out["input"]["bid_no"] == "R26BK01752065"
    assert out["base_amt"] == 184_521_490

    def missing(bid_no, client=None):
        raise LookupError("공고번호 X 를 찾지 못했습니다")
    monkeypatch.setattr(BN, "fetch_notice", missing)
    assert P.main(["--bid_no", "X", "--model", str(market_model)]) == 1
    assert "찾지 못했습니다" in capsys.readouterr().err


def test_cli_requires_est_price_or_bid_no(market_model):
    with pytest.raises(SystemExit):
        P.main(["--market", "--range", "3", "--model", str(market_model)])
