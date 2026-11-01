import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

import monitor_regime as MR


def test_previous_complete_month():
    assert MR.previous_complete_month(dt.date(2026, 10, 18)) == "2026-09"
    assert MR.previous_complete_month(dt.date(2026, 1, 5)) == "2025-12"


def test_range_stats_for_month(synthetic_market_df):
    stats = MR.range_stats_for_month(synthetic_market_df, "2024-02")
    assert set(stats.index).issubset({"±2", "±3"})
    assert list(stats.columns) == ["n", "mean", "std"]
    feb_rows = synthetic_market_df[
        pd.to_datetime(synthetic_market_df["contract_date"]).dt.strftime("%Y-%m") == "2024-02"
    ]
    assert stats["n"].sum() == len(feb_rows)

    empty_stats = MR.range_stats_for_month(synthetic_market_df, "2030-01")
    assert empty_stats.empty


def test_check_drift():
    # (a) 기준과 같은 값(n=1000)의 stats → 둘 다 빈 목록
    base_stats = pd.DataFrame(
        {"n": [1000, 1000], "mean": [99.985, 99.852], "std": [0.505, 0.740]},
        index=["±2", "±3"],
    )
    drift, warn = MR.check_drift(base_stats)
    assert drift == []
    assert warn == []

    # (b) ±3 mean 을 99.70 으로 → drift 1건, 메시지에 "평균" 과 "±3"
    b_stats = pd.DataFrame(
        {"n": [1000, 1000], "mean": [99.985, 99.700], "std": [0.505, 0.740]},
        index=["±2", "±3"],
    )
    drift, warn = MR.check_drift(b_stats)
    assert len(drift) == 1
    assert "평균" in drift[0] and "±3" in drift[0]
    assert warn == []

    # (c) ±2 std 를 0.60 으로 → drift 메시지에 "표준편차"
    c_stats = pd.DataFrame(
        {"n": [1000, 1000], "mean": [99.985, 99.852], "std": [0.600, 0.740]},
        index=["±2", "±3"],
    )
    drift, warn = MR.check_drift(c_stats)
    assert len(drift) == 1
    assert "표준편차" in drift[0] and "±2" in drift[0]
    assert warn == []

    # (d) n=100 → warnings 에 "보류", drift 없음
    d_stats = pd.DataFrame(
        {"n": [100, 100], "mean": [99.985, 99.852], "std": [0.505, 0.740]},
        index=["±2", "±3"],
    )
    drift, warn = MR.check_drift(d_stats)
    assert drift == []
    assert len(warn) == 2
    assert all("보류" in w for w in warn)

    # (e) stats 에 ±2 행이 없으면 warnings
    e_stats = pd.DataFrame(
        {"n": [1000], "mean": [99.852], "std": [0.740]},
        index=["±3"],
    )
    drift, warn = MR.check_drift(e_stats)
    assert drift == []
    assert len(warn) == 1
    assert "±2" in warn[0] and "보류" in warn[0]


def test_judge_backtest():
    ok_bt = {"overall": {"coverage_q25_q75": 0.50, "valid_q75_nomargin": 0.75}}
    assert MR.judge_backtest(ok_bt) == []

    bad_bt = {"overall": {"coverage_q25_q75": 0.40, "valid_q75_nomargin": 0.70}}
    msgs = MR.judge_backtest(bad_bt)
    assert len(msgs) == 2
    assert "커버리지" in msgs[0]
    assert "Q75 유효율" in msgs[1]


def test_collect_month(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.state = {"date": "", "calls_today": 10, "tasks": {}}

    client = FakeClient()
    saved = []
    monkeypatch.setattr(MR.cg, "save_state", lambda s, path=None: saved.append(s))

    # fake 가 정상 종료 → complete True, calls_today 10
    monkeypatch.setattr(MR.cg, "run_market", lambda *a, **k: None)
    res = MR.collect_month("2026-09", client=client)
    assert res["complete"] is True
    assert res["calls_today"] == 10
    assert len(saved) == 1

    # fake 가 MR.cg.DailyLimitReached("한도") 를 던지면 complete False 이고 reason 에 "한도"
    def fake_limit(*a, **k):
        raise MR.cg.DailyLimitReached("한도")

    monkeypatch.setattr(MR.cg, "run_market", fake_limit)
    res = MR.collect_month("2026-09", client=client)
    assert res["complete"] is False
    assert "한도" in res["reason"]

    # fake 가 MR.cg.ApiError("x") 를 던지면 그대로 전파(pytest.raises)
    def fake_api_err(*a, **k):
        raise MR.cg.ApiError("x")

    monkeypatch.setattr(MR.cg, "run_market", fake_api_err)
    with pytest.raises(MR.cg.ApiError):
        MR.collect_month("2026-09", client=client)

    # client 없이 호출하고 G2B_SERVICE_KEY 없으면 RuntimeError
    monkeypatch.delenv("G2B_SERVICE_KEY", raising=False)
    monkeypatch.setattr(MR, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        MR.collect_month("2026-09", client=None)


def test_write_report(tmp_path):
    stats = pd.DataFrame(
        {"n": [1000, 1000], "mean": [99.985, 99.852], "std": [0.505, 0.740]},
        index=["±2", "±3"],
    )
    bt = {
        "overall": {
            "n": 2000,
            "coverage_q25_q75": 0.50,
            "valid_q25_nomargin": 0.25,
            "valid_q50_nomargin": 0.50,
            "valid_q75_nomargin": 0.75,
            "ml_mae": 0.57,
            "baseline_mae": 0.57,
        }
    }

    # 이상 없음 케이스에 "이상 없음"
    p1 = tmp_path / "rep1.md"
    text1 = MR.write_report("2026-09", None, stats, [], [], bt, [], p1)
    assert p1.exists()
    assert "이상 없음" in text1

    # 변화 케이스에 "변화 감지" 와 메시지
    p2 = tmp_path / "rep2.md"
    text2 = MR.write_report("2026-09", None, stats, ["±3: 평균 변화"], [], bt, ["Q75 유효율 이상"], p2)
    assert p2.exists()
    assert "변화 감지" in text2
    assert "±3: 평균 변화" in text2
    assert "Q75 유효율 이상" in text2

    # 경고 케이스에 "## 경고"
    p3 = tmp_path / "rep3.md"
    text3 = MR.write_report("2026-09", None, stats, [], ["±2: 표본 부족"], bt, [], p3)
    assert p3.exists()
    assert "## 경고" in text3
    assert "±2: 표본 부족" in text3


def test_main(synthetic_market_df, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(MR, "MIN_N", 0)
    monkeypatch.setattr(MR.T, "load_market", lambda *a, **k: (synthetic_market_df, {}))
    fake_bt = {
        "overall": {
            "n": 100,
            "coverage_q25_q75": 0.50,
            "valid_q75_nomargin": 0.75,
            "valid_q25_nomargin": 0.25,
            "valid_q50_nomargin": 0.50,
            "ml_mae": 0.57,
            "baseline_mae": 0.57,
        }
    }
    monkeypatch.setattr(MR, "backtest_month", lambda ym: fake_bt)

    build_calls = []
    monkeypatch.setattr(MR.cg, "build", lambda *a, **k: build_calls.append(True))

    out_file = tmp_path / "r.md"

    # main(["--month", "2024-02", "--skip-collect", "--out", str(tmp_path / "r.md")]) → 0, 파일 존재, 출력에 "이상 없음", cg.build 호출됨
    code = MR.main(["--month", "2024-02", "--skip-collect", "--out", str(out_file)])
    captured = capsys.readouterr()
    assert code == 0
    assert out_file.exists()
    assert "이상 없음" in captured.out
    assert len(build_calls) == 1

    # --no-build 추가 시 cg.build 미호출
    build_calls.clear()
    code = MR.main(["--month", "2024-02", "--skip-collect", "--no-build", "--out", str(out_file)])
    assert code == 0
    assert len(build_calls) == 0

    # MR.REFERENCE 의 ±3 mean 을 90 으로 monkeypatch 하면 3
    new_ref = {
        "±2": {"mean": 99.985, "std": 0.505},
        "±3": {"mean": 90.0, "std": 0.740},
    }
    monkeypatch.setattr(MR, "REFERENCE", new_ref)
    code = MR.main(["--month", "2024-02", "--skip-collect", "--no-build", "--out", str(out_file)])
    assert code == 3

    # 원복
    monkeypatch.undo()
    monkeypatch.setattr(MR, "MIN_N", 0)
    monkeypatch.setattr(MR.T, "load_market", lambda *a, **k: (synthetic_market_df, {}))
    monkeypatch.setattr(MR, "backtest_month", lambda ym: fake_bt)
    monkeypatch.setattr(MR.cg, "build", lambda *a, **k: build_calls.append(True))

    # --month 2030-01 은 5
    code = MR.main(["--month", "2030-01", "--skip-collect", "--no-build", "--out", str(out_file)])
    assert code == 5

    # MR.collect_month 를 {"ym": ..., "complete": False, "reason": "한도", "calls_today": 950} 반환 함수로 바꾸고 --skip-collect 없이 실행하면 2
    monkeypatch.setattr(
        MR,
        "collect_month",
        lambda ym, **k: {"ym": ym, "complete": False, "reason": "한도", "calls_today": 950},
    )
    code = MR.main(["--month", "2024-02", "--no-build", "--out", str(out_file)])
    assert code == 2

    # --json 이면 stdout 이 JSON 으로 파싱되고 "exit_code" 와 "drift" 키가 있다
    capsys.readouterr()
    code = MR.main(["--month", "2024-02", "--skip-collect", "--no-build", "--json", "--out", str(out_file)])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "exit_code" in data
    assert "drift" in data
    assert data["exit_code"] == 0
