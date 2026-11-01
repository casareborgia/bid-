"""Phase 7: 실제값 검증 도구(evaluate), LLM 도구 함수(bid_tools), Ollama 루프(ollama_chat), MCP 서버 등록. 네트워크·Ollama 불필요."""
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bid_tools as B
import evaluate as E
import ollama_chat as OC
import predict_cli as P


# ---------------------------------------------------------------- evaluate: 단건
def test_compare_one_direction_and_validity(internal_model_path):
    art = P.load_model(internal_model_path)
    rep = B.predict_bid(1_000_000_000, work_type=2, bidder_count=25, model_path=str(internal_model_path), margin=0.1)
    q = {x["quantile"]: x for x in rep["quantiles"]}
    base = rep["base_amt"]
    # 실제 사정률이 Q75 보다 조금 높으면: 모든 기준 투찰가가 하한 미달 (마진 0.1% 보다 더 높게)
    actual = base * (q["Q75"]["yega_rate"] + 0.3) / 100
    res = E.compare_one(art, est_price=1e9, actual_yega_amt=actual, work_type=2, bidder_count=25, margin=0.1)
    assert res["position"] == "Q75 위" and not res["in_q25_q75"]
    assert [b["valid"] for b in res["bids"]] == [False, False, False]
    # 실제 사정률이 Q25~Q50 사이면: Q25 기준은 미달, Q50·Q75 기준은 유효
    mid = (q["Q25"]["yega_rate"] + q["Q50"]["yega_rate"]) / 2
    res = E.compare_one(art, est_price=1e9, actual_yega_amt=base * mid / 100, work_type=2, bidder_count=25, margin=0.0)
    assert res["position"] == "Q25~Q50" and res["in_q25_q75"]
    assert [b["valid"] for b in res["bids"]] == [False, True, True]
    assert res["bids"][2]["bid_minus_actual_lower"] > 0 > res["bids"][0]["bid_minus_actual_lower"]
    assert abs(res["actual"]["yega_rate"] - mid) < 1e-3 and res["abs_error_q50_pct"] > 0      # 사정률은 소수 4자리로 반올림


def test_evaluate_bid_no_from_parquet(tmp_path, monkeypatch, market_model_path):
    g = pd.DataFrame([
        {"bid_key": "R1|000|0|000", "bidNtceNo": "R1", "bid_name": "테스트 공사", "work_type": 1, "work_type_txt": "공사",
         "base_amt": 1.1e9, "yega_amt": 1.1e9 * 0.9985, "yega_rate": 99.85, "range_cls": "±3", "open_dt": pd.Timestamp("2024-03-01"),
         "bidder_count": 12.0, "dminsttNm": "기관A"},
        {"bid_key": "R2|000|0|000", "bidNtceNo": "R2", "bid_name": "범위미상", "work_type": 2, "work_type_txt": "용역",
         "base_amt": 5e8, "yega_amt": 5e8, "yega_rate": 100.0, "range_cls": None, "open_dt": pd.Timestamp("2024-03-02"),
         "bidder_count": np.nan, "dminsttNm": "기관B"},
    ])
    pq = tmp_path / "g2b.parquet"; g.to_parquet(pq, index=False)
    monkeypatch.setattr(E, "G2B_PATH", pq)
    art = P.load_model(market_model_path)
    out = E.evaluate_bid_no("R1", art)
    assert len(out) == 1 and out[0]["bid"]["range_cls"] == "±3" and out[0]["model_mode"] == "market"
    assert abs(out[0]["actual"]["yega_rate"] - 99.85) < 1e-3 and out[0]["position"] in ("Q25 아래", "Q25~Q50", "Q50~Q75", "Q75 위")
    out2 = E.evaluate_bid_no("R2", art)
    assert "skipped" in out2[0]
    with pytest.raises(LookupError):
        E.evaluate_bid_no("없음", art, live=False)
    monkeypatch.setattr(E, "fetch_actual", lambda bid_no, client=None: pd.DataFrame())   # 네트워크 차단
    with pytest.raises(LookupError, match="API"):
        E.evaluate_bid_no("없음", art)
    # bid_tools 경유: 오류는 dict 로
    r = B.evaluate_bid("없음", model_path=str(market_model_path))
    assert "error" in r
    r = B.evaluate_bid("R1", model_path=str(market_model_path))
    assert r["n_rows"] == 1 and r["results"][0]["bids"][2]["quantile"] == "Q75"


# ---------------------------------------------------------------- evaluate: 배치
def test_expand_months():
    assert E.expand_months("2026-09") == ["2026-09"]
    assert E.expand_months("2026-01,2026-03") == ["2026-01", "2026-03"]
    assert E.expand_months("2025-11~2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_backtest_market_and_internal_synthetic(synthetic_market_df, synthetic_internal_df):
    res = E.backtest("market", ["2024-07"], df_override=synthetic_market_df, sample=5, margin=0.1)
    o = res["overall"]
    assert res["n_test"] > 0 and res["n_train"] > 100 and res["cutoff"] == "2024-07-01"
    assert set(res["by_group"]) <= {"±2", "±3"} and len(res["sample"]) == 5
    assert 0.3 < o["coverage_q25_q75"] < 0.7
    assert o["valid_q25"] < o["valid_q50"] < o["valid_q75"]                 # 통과확률 방향
    assert o["valid_q75"] >= o["valid_q75_nomargin"]                       # 마진은 유효율을 높인다
    assert abs(o["valid_q75"] - o["valid_q25"] - o["coverage_q25_q75"]) < 0.15
    res_i = E.backtest("internal", ["2024-03", "2024-04"], df_override=synthetic_internal_df, sample=3)
    assert set(res_i["by_group"]) <= {"공사", "용역"} and res_i["n_test"] > 0
    assert all(r["group"] in ("공사", "용역") for r in res_i["sample"])
    with pytest.raises(ValueError):                                        # 홀드아웃에 행이 없으면
        E.backtest("market", ["2030-01"], df_override=synthetic_market_df)


def test_write_backtest_report(tmp_path, synthetic_market_df):
    res = E.backtest("market", ["2024-07"], df_override=synthetic_market_df, sample=3)
    text = E.write_backtest_report(res, tmp_path / "bt.md")
    assert "## 전체" in text and "## 예가 범위별" in text and "## 판정" in text and "표본 3건" in text


# ---------------------------------------------------------------- bid_tools
def test_predict_bid_tool_paths_and_errors(internal_model_path, market_model_path):
    r = B.predict_bid(1e9, work_type=1, market=True, range_pct=3, model_path=str(market_model_path))
    assert r["model"]["mode"] == "market" and r["recommended"]["basis"] == "Q75" and "note" in r
    assert B.predict_bid(1e9, market=True, model_path=str(market_model_path))["error"].startswith("시장 모델은 range_pct")
    assert "error" in B.predict_bid(1e9, range_pct=3, model_path=str(internal_model_path))
    assert "error" in B.predict_bid(1e9, basis="Q90", model_path=str(internal_model_path))
    assert "error" in B.predict_bid(1e9, market=True, range_pct=3, model_path=str(internal_model_path))   # 시장 모델 아님
    r = B.predict_bid(1e9, negotiated=True, model_path=str(internal_model_path))
    assert r["model"]["mode"] == "negotiated"
    r = B.compare_manual(1e9, 1.098e9, work_type=1, market=True, range_pct=3, model_path=str(market_model_path))
    assert r["position"] in ("Q25 아래", "Q25~Q50", "Q50~Q75", "Q75 위") and len(r["bids"]) == 3
    info = B.model_info(internal_path=str(internal_model_path), market_path=str(market_model_path))
    assert info["internal"]["negotiated_dist"]["n"] == 40 and set(info["market"]["priors_by_range"]) == {"±2", "±3"}
    assert "rules" in info and "error" in B.model_info(internal_path="/없는/경로.joblib")["internal"]


class _FixtureClient:
    """op → tests/fixtures/g2b 녹화본. 없는 op 는 빈 응답."""
    def __init__(self, mapping): self.mapping, self.calls = mapping, []
    def get(self, op, params):
        from collect_g2b_api import parse_body
        self.calls.append((op, params))
        name = self.mapping.get(op, "notice_cnstwk_empty")
        return parse_body(json.loads((Path(__file__).parent / "fixtures" / "g2b" / f"{name}.json").read_text()))


def test_evaluate_bid_no_live_api(tmp_path, monkeypatch, market_model_path):
    """수집 parquet 에 없는 공고는 낙찰정보 API(실제값)와 입찰공고 API(추정가격·하한율·A값)로 비교한다 (녹화본)."""
    pq = tmp_path / "g2b.parquet"
    pd.DataFrame({"bid_key": ["X"], "bidNtceNo": ["X"]}).to_parquet(pq, index=False)
    monkeypatch.setattr(E, "G2B_PATH", pq)
    actual = _FixtureClient({"getOpengResultListInfoCnstwkPreparPcDetail": "actual_detail_cnstwk_R26BK01752065",
                             "getOpengResultListInfoCnstwk": "actual_list_cnstwk_R26BK01752065"})
    notice = _FixtureClient({"getBidPblancListInfoCnstwk": "notice_cnstwk_R26BK01752065",
                             "getBidPblancListInfoCnstwkBsisAmount": "bsis_cnstwk_R26BK01752065"})
    art = P.load_model(market_model_path)
    out = E.evaluate_bid_no("R26BK01752065", art, client=actual, notice_client=notice)
    assert actual.calls[0][1]["inqryDiv"] == 2 and actual.calls[1][1]["inqryDiv"] == 4
    r = out[0]
    assert r["bid"]["source"] == "api" and r["bid"]["notice_used"] and r["bid"]["range_cls"] == "±3"
    assert r["bid"]["bidder_count"] == 1601
    assert r["base_amt"] == 184_521_490 and r["actual"]["yega_amt"] == 184_804_325
    assert r["input"]["lower_limit_pct"] == 89.745 and r["input"]["a_value"] == 11_684_881
    assert r["actual"]["lower_limit_amt"] == round(P.lower_limit_amount(184_804_325, 89.745, 11_684_881))


def test_predict_bid_tool_with_bid_no(market_model_path, monkeypatch):
    import bid_notice as BN
    notice = {"bid_no": "R26BK01752065", "bid_ord": "000", "name": "전기공사", "work_type": 1, "notice_date": "2026-10-01",
              "agency": "서울특별시", "est_price": 167_746_809.0, "base_amt": 184_521_490.0, "range_pct": 3,
              "lower_limit_pct": 89.745, "a_value": 11_684_881.0, "a_items": {}, "servc_kind": None,
              "award_method_name": "적격심사", "contract_method": "제한경쟁", "open_dt": "2026-10-13 13:00:00", "warnings": []}
    monkeypatch.setattr(BN, "fetch_notice", lambda bid_no, client=None: dict(notice))
    r = B.predict_bid(bid_no="R26BK01752065", model_path=str(market_model_path))
    assert r["model"]["mode"] == "market" and r["base_amt"] == 184_521_490
    assert r["input"]["a_value"] == 11_684_881 and r["input"]["lower_limit_source"] == "notice"
    assert "error" in B.predict_bid(bid_no="R26BK01752065", work_type=2, a_value=1_000_000, model_path=str(market_model_path))

    def missing(bid_no, client=None):
        raise LookupError("공고번호 X 를 찾지 못했습니다")
    monkeypatch.setattr(BN, "fetch_notice", missing)
    assert "찾지 못했습니다" in B.predict_bid(bid_no="X", model_path=str(market_model_path))["error"]
    assert "bid_no" in B.predict_bid(model_path=str(market_model_path))["error"]


# ---------------------------------------------------------------- ollama_chat: 도구 호출 루프 (가짜 클라이언트)
class _Fn:
    def __init__(self, name, arguments): self.name, self.arguments = name, arguments
class _Call:
    def __init__(self, name, arguments): self.function = _Fn(name, arguments)
class _Msg:
    def __init__(self, content="", tool_calls=None): self.content, self.tool_calls, self.role = content, tool_calls, "assistant"
class _Resp:
    def __init__(self, msg): self.message = msg


class FakeClient:
    """1차: 문자열 인자로 predict_bid 호출 요구 → 2차: 도구 결과를 받아 최종 답변."""
    def __init__(self): self.calls = []
    def chat(self, model, messages, tools=None, think=None):
        self.calls.append([m if isinstance(m, dict) else {"role": "assistant"} for m in messages])
        if not any(isinstance(m, dict) and m.get("role") == "tool" for m in messages):
            return _Resp(_Msg(tool_calls=[_Call("predict_bid", {"est_price": "1,000,000,000원", "work_type": "1", "market": "true", "range_pct": "3"})]))
        tool_msg = next(m for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        rec = json.loads(tool_msg["content"])["recommended"]
        return _Resp(_Msg(content=f"권장 투찰금액 {rec['bid_amt']:,}원 ({rec['basis']}, 통과확률 {int(rec['pass_prob']*100)}%)"))


def test_run_turn_dispatches_tools_with_coercion(market_model_path):
    def predict_bid(est_price: float, work_type: int = 2, market: bool = False, range_pct: int | None = None, **kw):
        return B.predict_bid(est_price, work_type=work_type, market=market, range_pct=range_pct, model_path=str(market_model_path))
    client = FakeClient()
    messages = [{"role": "system", "content": OC.SYSTEM_PROMPT}, {"role": "user", "content": "10억 공사 ±3% 투찰가"}]
    answer = OC.run_turn(client, "fake", messages, tools=[predict_bid], dispatch={"predict_bid": predict_bid})
    assert "권장 투찰금액" in answer and "Q75" in answer and "75%" in answer
    roles = [m["role"] if isinstance(m, dict) else "assistant" for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_payload = json.loads(messages[3]["content"])
    assert tool_payload["input"]["range_cls"] == "±3" and tool_payload["input"]["work_type"] == 1 and tool_payload["input"]["market"] is True


def test_run_turn_unknown_tool_returns_error_to_model():
    class C:
        n = 0
        def chat(self, model, messages, tools=None, think=None):
            self.n += 1
            if self.n == 1:
                return _Resp(_Msg(tool_calls=[_Call("nope", {})]))
            return _Resp(_Msg(content="끝"))
    messages = [{"role": "user", "content": "x"}]
    assert OC.run_turn(C(), "fake", messages, tools=[], dispatch={}) == "끝"
    assert "알 수 없는 도구" in messages[2]["content"]


def test_coerce_types():
    def f(a: float, b: int = 1, c: bool = False, d: str | None = None, e: int | None = None): ...
    out = OC._coerce(f, {"a": "1,000원", "b": "3.0", "c": "예", "d": 5, "e": "", "zzz": 1})
    assert out == {"a": 1000.0, "b": 3, "c": True, "d": 5, "e": None}


# ---------------------------------------------------------------- MCP 서버
def test_mcp_server_registers_all_tools():
    import mcp_server as M
    names = {t.name for t in asyncio.run(M.server.list_tools())}
    assert names == {f.__name__ for f in B.TOOLS}
