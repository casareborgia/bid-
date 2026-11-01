"""규칙표 vs 입찰공고 대량 검증(check_notice_rules) 테스트. 네트워크 없이 합성 원시 행으로 조인·판정·요약을 확인한다."""
import json

import pandas as pd
import pytest

import check_notice_rules as C


def _notice(no, agency, est, lower, method, ord_="000", date="2026-10-05 10:00:00", kind=None):
    r = {"bidNtceNo": no, "bidNtceOrd": ord_, "bidNtceDt": date, "dminsttNm": agency, "ntceInsttNm": agency,
         "presmptPrce": str(est), "sucsfbidLwltRate": str(lower), "sucsfbidMthdNm": method}
    if kind:
        r["srvceDivNm"] = kind
    return r


def _bsis(no, base, rng, ord_="000"):
    return {"bidNtceNo": no, "bidNtceOrd": ord_, "bssamt": str(base),
            "rsrvtnPrceRngBgnRate": f"-{rng}", "rsrvtnPrceRngEndRate": f"+{rng}"}


RAW = {
    (1, "notice"): [
        _notice("C1", "서울특별시", 150_000_000, 89.745, "적격심사제-추정가격이 2억원 미만"),             # 지자체 공사 10억 미만 → 일치
        _notice("C2", "서울특별시 강남구", 150_000_000, 87.745, "적격심사제", ord_="000"),
        _notice("C2", "서울특별시 강남구", 150_000_000, 88.000, "적격심사제", ord_="001"),             # 최신 차수만 → 불일치
        _notice("C3", "어떤주식회사", 150_000_000, 89.745, "적격심사제"),                              # 규칙표 판정 불가
    ],
    (1, "bsis"): [_bsis("C1", 165_000_000, 3), _bsis("C2", 165_000_000, 3, ord_="001"), _bsis("C3", 1, 2)],
    (2, "notice"): [
        _notice("S1", "충청남도 공주시", 62_200_000, 88, "소액수의견적-소액수의견적", kind="기술용역"),   # 소액수의 → 규칙표와 다름
    ],
    (2, "bsis"): [_bsis("S1", 68_420_000, 3)],
}


def test_method_category():
    assert C.method_category("적격심사제-추정가격이 2억원 미만") == "적격심사"
    assert C.method_category("소액수의견적-소액수의견적(2인 이상 견적 제출)") == "소액수의견적"
    assert C.method_category("제한적최저가") == "제한적최저가"
    assert C.method_category(None) == "기타"


def test_build_table_and_summary():
    df = C.build_table(RAW)
    assert sorted(df["bidNtceNo"]) == ["C1", "C2", "C3", "S1"]                 # 정정공고는 최신 차수 1건
    c2 = df[df["bidNtceNo"] == "C2"].iloc[0]
    assert c2["bidNtceOrd"] == "001" and c2["notice_lower"] == 88.0 and c2["notice_range"] == 3
    s1 = df[df["bidNtceNo"] == "S1"].iloc[0]
    assert s1["servc_kind"] == "tech" and s1["method"] == "소액수의견적" and s1["rule_lower"] == 87.745
    assert pd.isna(df[df["bidNtceNo"] == "C3"].iloc[0]["rule_lower"])

    s = C.summarize(df)
    assert s["n_notice"] == 4 and s["n_judged"] == 3
    assert s["n_qualified"] == 2 and s["match_rate_qualified"] == pytest.approx(0.5)
    assert s["mismatch_pairs"][0][1] == 1                                       # 강남구 88.0 1건
    assert s["unjudged_qualified"] == 1
    assert s["by_method"].loc["소액수의견적", "match_rate"] == 0
    assert s["n_range"] == 3 and s["range_match_rate"] == pytest.approx(1.0)     # C3(±2)은 규칙표 판정 불가라 제외


def test_fetch_month_uses_cache_and_requires_fetch(tmp_path):
    for (wt, kind), op in C.OPS.items():
        rows = RAW.get((wt, kind), [])
        (tmp_path / f"{op}_2026-10.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    raw = C.fetch_month("2026-10", cache_dir=tmp_path, fetch=False)
    assert len(raw[(1, "notice")]) == 4
    with pytest.raises(FileNotFoundError):
        C.fetch_month("2026-09", cache_dir=tmp_path, fetch=False)


def test_fetch_month_pages_and_drops_personal_fields(tmp_path):
    class Pager:
        def __init__(self): self.calls = []
        def get(self, op, params):
            self.calls.append((op, params["pageNo"]))
            item = dict(_notice("N", "서울특별시", 1, 89.745, "적격심사"), ntceInsttOfclNm="홍길동", ntceInsttOfclTelNo="02-0000")
            return {"items": [item], "totalCount": 1000 if params["pageNo"] == 1 else 1000}
    p = Pager()
    raw = C.fetch_month("2026-10", client=p, cache_dir=tmp_path)
    assert [pg for _, pg in p.calls][:2] == [1, 2] and len(p.calls) == 8      # 4개 오퍼레이션 × 2페이지
    text = "".join(f.read_text() for f in tmp_path.glob("*.jsonl"))
    assert "홍길동" not in text and "02-0000" not in text
    assert len(raw[(1, "notice")]) == 2
