"""투찰 전략 재연 백테스트·업체별 개찰 수집(sim.backtest_strategies, sim.collect_bidders) 테스트. 합성 데이터만 쓴다."""
import numpy as np
import pandas as pd

from sim import backtest_strategies as B
from sim import collect_bidders as CB


def _auctions(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rc = "±3" if i % 3 else "±2"
        y = rng.normal(99.86 if rc == "±3" else 99.99, 0.74 if rc == "±3" else 0.51)
        k = int(rng.integers(3, 40))
        rows.append({"bid_key": f"B{i}", "range_cls": rc, "y": y, "n": k, "open_dt": pd.Timestamp("2026-10-01") + pd.Timedelta(hours=i),
                     "rates": rng.normal(100.1, 0.6, k)})                # 경쟁사는 실제 사정률보다 높게 투찰(실측 경향)
    return pd.DataFrame(rows)


def test_evaluate_strategies_and_report(tmp_path):
    a = _auctions()
    hist = {"±2": np.random.default_rng(1).normal(99.99, 0.51, 5000), "±3": np.random.default_rng(2).normal(99.86, 0.74, 5000)}
    res, meta = B.evaluate(a, y_samples_n=1000, y_history=hist)
    assert meta["n_train"] == 60 and meta["n_test"] == 60 and meta["y_source"] == "과거 시장 데이터"
    for s in ("Q25", "Q50", "Q75", "MC 최적", "MC 재표본"):
        assert res[s].dtype == bool and res[f"{s}_valid"].dtype == bool
    # 통과율은 분위수 순서를 따른다
    assert res["Q25_valid"].mean() <= res["Q50_valid"].mean() <= res["Q75_valid"].mean()
    text = B.write_report(res, meta, 0.99, tmp_path / "r.md")
    assert "MC 최적" in text and "99.00%" in text


def test_anonymize_drops_names_and_hashes_bizno():
    item = {"bidNtceNo": "R1", "bidprcAmt": "100", "prcbdrNm": "회사", "prcbdrCeoNm": "홍길동", "prcbdrBizno": "1234567890",
            "drwtNo1": " 07", "rmrk": "정상"}
    row = CB.anonymize(item)
    assert "prcbdrNm" not in row and "prcbdrCeoNm" not in row and "prcbdrBizno" not in row
    assert row["bidder"] == CB.anonymize(dict(item))["bidder"] and len(row["bidder"]) == 12 and row["bidder"] != "1234567890"
