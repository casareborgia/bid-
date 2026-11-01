"""Phase 2 수집기 오프라인 테스트 (네트워크 없음)."""
import json

import pandas as pd
import pytest

import collect_g2b_api as cg


def test_month_range_and_bounds():
    assert cg.month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert cg.month_bounds("2026-02") == ("202602010000", "202602282359")
    assert cg.month_bounds("2024-02") == ("202402010000", "202402292359")


def test_parse_body_ok_and_errors():
    body = cg.parse_body({"response": {"header": {"resultCode": "00"}, "body": {"items": [{"a": 1}], "totalCount": "1"}}})
    assert body["items"] == [{"a": 1}]
    body = cg.parse_body({"response": {"header": {"resultCode": "00"}, "body": {"items": "", "totalCount": "0"}}})
    assert body["items"] == []
    with pytest.raises(cg.ApiError, match="resultCode=07"):
        cg.parse_body({"nkoneps.com.response.ResponseError": {"header": {"resultCode": "07", "resultMsg": "입력범위값 초과 에러"}}})
    with pytest.raises(cg.ApiError, match="resultCode=30"):
        cg.parse_body({"response": {"header": {"resultCode": "30", "resultMsg": "등록되지 않은 서비스키"}}})


def _detail_row(no, sno, bssamt, plnprc, tot="15", ord_="000", clsfc="0", rbid="000", bsis=""):
    return {"bidNtceNo": no, "bidNtceOrd": ord_, "bidClsfcNo": clsfc, "rbidNo": rbid, "bidNtceNm": f"공고 {no}",
            "plnprc": plnprc, "bssamt": bssamt, "totRsrvtnPrceNum": tot, "compnoRsrvtnPrceSno": sno, "bsisPlnprc": bsis,
            "bssamtBssUpNum": "8", "rlOpengDt": "2026-09-04 13:58:14", "_work": "servc", "_ym": "2026-09", "_page": 1}


def test_build_detail_dedupes_and_joins():
    # R1: 15개 예비가격이 기초금액 ±3% 안에 균등 → half_range 3.0 → "±3"
    bsis = [str(int(406000000 * (1 + d / 100))) for d in [-3.0 + 6.0 * i / 14 for i in range(15)]]
    detail = pd.DataFrame(
        [_detail_row("R1", str(i), "406000000", "407424050", bsis=bsis[i - 1]) for i in range(1, 16)]   # 복수예가 15행 → 1건
        + [_detail_row("R2", "", "", "54600000", tot="1")]                            # 단일예가(bssamt 공란) → 제외
        + [_detail_row("R3", "1", "100000000", "99500000"), _detail_row("R3", "1", "100000000", "99500000", clsfc="1")]  # 분류번호 다르면 별건
    )
    listing = pd.DataFrame([
        {"bidNtceNo": "R1", "bidNtceOrd": "000", "bidClsfcNo": "0", "rbidNo": "000", "opengDt": "2026-09-04 14:00:00",
         "prtcptCnum": "12", "dminsttNm": "기관A", "progrsDivCdNm": "개찰완료"},
    ])
    out = cg.build_detail(detail, listing)
    assert len(out) == 3
    r1 = out[out["bidNtceNo"] == "R1"].iloc[0]
    assert r1["bidder_count"] == 12 and r1["dminsttNm"] == "기관A" and r1["work_type"] == 2
    assert abs(r1["yega_rate"] - 100.3508) < 0.001
    assert str(r1["open_dt"]).startswith("2026-09-04 14:00")
    assert abs(r1["half_range"] - 3.0) < 0.01 and r1["range_cls"] == "±3" and r1["n_prices"] == 15
    r3 = out[out["bidNtceNo"] == "R3"]
    assert len(r3) == 2 and r3["bidder_count"].isna().all()
    assert r3["range_cls"].isna().all()                                   # 예비가격이 없으면 범위 미상
    assert str(r3.iloc[0]["open_dt"]).startswith("2026-09-04 13:58")     # 목록 미매칭 시 rlOpengDt 대체
    assert (out["source"] == "g2b").all()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, json.dumps(payload)
    def json(self):
        return self._p


def _page_payload(items, total):
    return {"response": {"header": {"resultCode": "00", "resultMsg": "정상"}, "body": {"items": items, "totalCount": total, "numOfRows": 999, "pageNo": 1}}}


def test_crawl_pages_resumes_and_respects_daily_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "SLEEP_SEC", 0)
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["pageNo"])
        items = [{"bidNtceNo": f"R{params['pageNo']}", "bidNtceOrd": "000", "bidClsfcNo": "0", "rbidNo": "000"}]
        return FakeResponse(_page_payload(items, total=3 * cg.PAGE_SIZE))   # 3페이지짜리 태스크

    state = {"date": "", "calls_today": 0, "tasks": {}}
    client = cg.G2BClient("k", state, daily_limit=2)
    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr(cg, "save_state", lambda s, path=None: None)
    raw = tmp_path / "d.jsonl"

    with pytest.raises(cg.DailyLimitReached):
        cg.crawl_pages(client, "op", {}, "t1", raw, {"_work": "servc"})
    assert calls == [1, 2] and state["tasks"]["t1"]["next_page"] == 3 and not state["tasks"]["t1"]["complete"]

    # 다음 날: 한도 리셋 후 이어받기
    state["date"] = "1999-01-01"
    client.daily_limit = 10
    t = cg.crawl_pages(client, "op", {}, "t1", raw, {"_work": "servc"})
    assert calls == [1, 2, 3] and t["complete"] and t["rows"] == 3
    assert len(raw.read_text().splitlines()) == 3
    # 완료된 태스크는 재호출하지 않음
    cg.crawl_pages(client, "op", {}, "t1", raw, {"_work": "servc"})
    assert calls == [1, 2, 3]


def test_network_error_message_does_not_leak_service_key(monkeypatch):
    """접속 타임아웃 등 requests 예외 메시지에는 요청 URL(ServiceKey 포함)이 들어간다. 최종 ApiError 에는 키가 없어야 한다."""
    import requests
    key = "adc9b33b7d0f9d37SECRETKEY0a6"
    monkeypatch.setattr(cg, "SLEEP_SEC", 0)
    monkeypatch.setattr(cg.time, "sleep", lambda *_: None)          # 재시도 백오프 생략
    client = cg.G2BClient(key, {"date": "", "calls_today": 0, "tasks": {}})

    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectTimeout(
            f"HTTPSConnectionPool(host='apis.data.go.kr', port=443): Max retries exceeded with url: "
            f"/1230000/as/ScsbidInfoService/op?ServiceKey={key}&type=json&pageNo=1 (Caused by ConnectTimeoutError)")
    monkeypatch.setattr(client.session, "get", fake_get)

    with pytest.raises(cg.ApiError) as ei:
        client.get("op", {"pageNo": 1})
    msg = str(ei.value)
    assert key not in msg and "ServiceKey=***" in msg and "재시도 3회 실패" in msg

    # resultCode 오류(즉시 중단)도 마찬가지로 키 없이 전달된다
    def fake_get_auth(url, params=None, timeout=None):
        return FakeResponse({"response": {"header": {"resultCode": "30", "resultMsg": f"등록되지 않은 키 {key}"}}})
    monkeypatch.setattr(client.session, "get", fake_get_auth)
    with pytest.raises(cg.ApiError, match="resultCode=30") as ei:
        client.get("op", {"pageNo": 1})
    assert key not in str(ei.value)
