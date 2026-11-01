"""Phase 8-C, 8-D: 예가 범위·낙찰하한율 규칙 엔진(src/rules.py), CLI --agency, bid_tools 통합 테스트."""
import json
import pandas as pd
import pytest

import bid_tools as B
import predict_cli as P
import rules as R


def test_resolve_range_verified_and_unverified():
    # 1. 남동발전 -> ±2
    r1 = R.resolve_range("한국남동발전(주)", work_type=1)
    assert r1 is not None and r1["range_pct"] == 2 and r1["range_cls"] == "±2" and r1["status"] == "verified"
    assert "제34조" in r1["basis_text"]

    # 2. 한전 및 주요 SOC 공기업 -> ±2
    for ag in ("한국전력공사", "한국도로공사", "한국토지주택공사", "한국수자원공사", "국가철도공단"):
        r = R.resolve_range(ag, work_type=1)
        assert r is not None and r["range_pct"] == 2

    # 3. 지자체 -> ±3
    for ag in ("서울특별시 강남구", "경기도 수원시", "부산광역시", "전라남도 해남군", "서울특별시 상수도사업본부"):
        r = R.resolve_range(ag, work_type=1)
        assert r is not None and r["range_pct"] == 3 and r["range_cls"] == "±3"

    # 4. 교육청/초중고 -> ±3, 국립대학교 -> None (unverified)
    r_edu = R.resolve_range("경기도교육청 경기도수원교육지원청", work_type=2)
    assert r_edu is not None and r_edu["range_pct"] == 3
    assert R.resolve_range("서울대학교", work_type=1) is None
    r_univ_unv = R.resolve_range("서울대학교", work_type=1, include_unverified=True)
    assert r_univ_unv is not None and r_univ_unv["status"] == "unverified"

    # 5. 미등록 기관 또는 기간 밖 -> None
    assert R.resolve_range("알수없는민간기관", work_type=1) is None
    assert R.resolve_range("한국남동발전", work_type=1, as_of="2015-01-01") is None


def test_resolve_lower_limit_by_agency_and_price():
    """공사 하한율은 조달청 2026-01-30 공고분·지자체 2025-07-01부터 2%p 상향 (사용자 확인 2026-11-01)."""
    def pct(ag, wt, price, as_of):
        r = R.resolve_lower_limit(ag, work_type=wt, est_price=price, as_of=as_of)
        return None if r is None else r["lower_limit_pct"]

    # 남동발전 공사 (조달청 시설공사 적격심사세부기준 준용): 10억 미만 / 10~50억 / 50~100억
    assert pct("한국남동발전", 1, 500_000_000, "2026-11-01") == pytest.approx(89.745)
    assert pct("한국남동발전", 1, 2_000_000_000, "2026-11-01") == pytest.approx(88.745)
    assert pct("한국남동발전", 1, 7_000_000_000, "2026-11-01") == pytest.approx(87.495)
    assert pct("한국남동발전", 1, 12_000_000_000, "2026-11-01") is None  # 100억 이상은 종합심사
    # 개정 전 공고(2026-01-29)는 이전 하한율
    assert pct("한국남동발전", 1, 500_000_000, "2026-01-29") == pytest.approx(87.745)
    assert pct("한국남동발전", 1, 2_000_000_000, "2026-01-29") == pytest.approx(86.745)

    # 지자체 공사는 2025-07-01부터 상향. 50억 이상은 공고마다 달라 판정 불가 (2026-10 공고 실측)
    assert pct("서울특별시 강남구", 1, 500_000_000, "2025-07-01") == pytest.approx(89.745)
    assert pct("서울특별시 강남구", 1, 500_000_000, "2025-06-30") == pytest.approx(87.745)
    assert pct("서울특별시 강남구", 1, 2_000_000_000, "2026-11-01") == pytest.approx(88.745)
    assert pct("서울특별시 강남구", 1, 7_000_000_000, "2026-11-01") is None
    assert pct("서울특별시 강남구", 1, 20_000_000_000, "2026-11-01") is None
    assert pct("서울특별시 강남구", 2, 2_000_000_000, "2026-11-01") is None  # 용역 10억 이상, 종류 미지정 → 일반이 판정 불가라 None

    # 미등록 기관 -> None
    assert R.resolve_lower_limit("미등록기관", work_type=1, est_price=500_000_000) is None


def test_predict_cli_internal_lower_limit_defaults_to_koen_rule(capsys):
    """사내 모델은 --lower_limit 생략 시 한국남동발전 규칙(조달청 기준 준용)을 쓰고, 명시값이 우선한다."""
    import json
    import predict_cli as P
    base = ["--est_price", "500000000", "--work_type", "1", "--as_of", "2026-10-01", "--json"]
    assert P.main(base) == 0
    i = json.loads(capsys.readouterr().out)["input"]
    assert i["lower_limit_pct"] == pytest.approx(89.745) and i["lower_limit_source"] == "auto"
    assert P.main(base + ["--lower_limit", "87.745"]) == 0
    i = json.loads(capsys.readouterr().out)["input"]
    assert i["lower_limit_pct"] == pytest.approx(87.745) and i["lower_limit_source"] == "explicit"


def test_check_regime_rules_against_synthetic_data(tmp_path):
    df = pd.DataFrame([
        {"dminsttNm": "한국남동발전", "work_type": 1, "range_cls": "±2", "yega_rate": 100.0},
        {"dminsttNm": "한국도로공사", "work_type": 1, "range_cls": "±2", "yega_rate": 99.9},
        {"dminsttNm": "경기도 수원시", "work_type": 1, "range_cls": "±3", "yega_rate": 99.8},
        {"dminsttNm": "경기도교육청", "work_type": 2, "range_cls": "±3", "yega_rate": 99.7},
        {"dminsttNm": "기타미분류기관", "work_type": 2, "range_cls": "±2", "yega_rate": 100.1},
    ])
    pq = tmp_path / "g2b.parquet"
    out_md = tmp_path / "check.md"
    df.to_parquet(pq, index=False)
    res = R.check_regime_rules_against_data(parquet_path=pq, out_path=out_md)
    assert out_md.is_file()
    verified = res[res["status"] == "verified"]
    assert (verified["match_rate"] == 1.0).all()


def test_predict_cli_agency_auto_and_explicit_override(capsys, market_model_path):
    # (a) 지자체 자동 판정 -> ±3
    rc = P.main(["--est_price", "1000000000", "--work_type", "1", "--market", "--agency", "서울특별시 강남구",
                 "--model", str(market_model_path), "--json"])
    assert rc == 0
    out_a = json.loads(capsys.readouterr().out)
    assert out_a["input"]["range_cls"] == "±3" and out_a["input"]["range_source"] == "auto"
    assert out_a["input"]["range_basis"]

    # (b) 주요 공기업 자동 판정 -> ±2
    rc = P.main(["--est_price", "1000000000", "--work_type", "1", "--market", "--agency", "한국도로공사",
                 "--model", str(market_model_path), "--json"])
    assert rc == 0
    out_b = json.loads(capsys.readouterr().out)
    assert out_b["input"]["range_cls"] == "±2" and out_b["input"]["range_source"] == "auto"

    # (c) 명시값(--range 2)이 자동 판정(지자체 ±3)보다 우선
    rc = P.main(["--est_price", "1000000000", "--work_type", "1", "--market", "--agency", "서울특별시 강남구",
                 "--range", "2", "--model", str(market_model_path), "--json"])
    assert rc == 0
    out_c = json.loads(capsys.readouterr().out)
    assert out_c["input"]["range_cls"] == "±2" and out_c["input"]["range_source"] == "explicit"

    # (d) 미확인 기관 + --range 미지정 -> 종료 코드 1
    rc = P.main(["--est_price", "1000000000", "--work_type", "1", "--market", "--agency", "알수없는기관",
                 "--model", str(market_model_path), "--json"])
    assert rc == 1
    capsys.readouterr()


def test_bid_tools_agency_and_rule_basis(market_model_path):
    r1 = B.predict_bid_simple(est_price=1_000_000_000, market=True, work_type=1, agency="경기도 수원시")
    assert "error" not in r1 and r1["input"]["range_cls"] == "±3" and "근거:" in r1["note"]

    r2 = B.rule_basis(agency="한국남동발전", work_type=1, est_price=500_000_000, as_of="2026-11-01")
    assert "error" not in r2 and r2["range_rule"]["range_pct"] == 2
    assert r2["lower_limit_rule"]["lower_limit_pct"] == pytest.approx(89.745)

    r3 = B.predict_bid(est_price=1_000_000_000, market=True, work_type=1, agency="미등록기관", model_path=str(market_model_path))
    assert "error" in r3

    assert B.rule_basis("")["error"]


@pytest.mark.parametrize(
    "agency,expected_pct",
    [
        ("한국남동발전 영흥발전본부", 2),
        ("한국전력공사 남서울본부", 2),
        ("한국토지주택공사", 2),
        ("한국도로공사", 2),
        ("한국수자원공사", 2),
        ("국가철도공단", 2),
        ("서울특별시 강남구", 3),
        ("경기도 수원시", 3),
        ("부산광역시 상수도사업본부", 3),
        ("경기도교육청", 3),
        ("서울특별시강남서초교육지원청", 3),
        ("서울대학교", None),
        ("조달청", None),
    ],
)
def test_resolve_range_parametrized(agency, expected_pct):
    res = R.resolve_range(agency, work_type=1)
    if expected_pct is None:
        assert res is None
    else:
        assert res is not None and res["range_pct"] == expected_pct


def test_lower_limit_local_scope_follows_regime_classification():
    """지자체 낙찰하한율 행은 예가 범위 규칙표의 기관 분류를 따른다 — 지역명으로 시작하는 대학·공사·과기원에 지자체 예규를 붙이지 않는다."""
    for ag in ["부산대학교", "인천국제공항공사", "서울주택도시공사", "경기도시공사", "광주과학기술원",
               "중소벤처기업부 부산기계공업고등학교"]:
        assert R.resolve_range(ag, work_type=1) is None, ag
        assert R.resolve_lower_limit(ag, work_type=1, est_price=2_000_000_000) is None, ag
    edu = R.resolve_lower_limit("대구광역시교육청", work_type=1, est_price=2_000_000_000, as_of="2026-11-01")
    assert edu["rule_id"] == "local_gov_cnstwk_1b_5b" and "2100000281396" in edu["basis_text"]


def test_resolve_lower_limit_koen_service_general_and_tech():
    """남동발전 용역은 조달청 기술용역 기준 준용(2026-05-26 2%p 상향, 구간 2억/5억/10억 — 2026-10 공고 실측).
    조달청 일반용역은 별표별로 하한율이 달라 판정 불가."""
    def pct(price, as_of, kind):
        r = R.resolve_lower_limit("한국남동발전", work_type=2, est_price=price, as_of=as_of, servc_kind=kind)
        return None if r is None else r["lower_limit_pct"]

    prices = [50_000_000, 150_000_000, 300_000_000, 700_000_000, 2_000_000_000]
    assert [pct(p, "2026-11-01", "tech") for p in prices] == pytest.approx([89.745, 89.745, 88.745, 87.495, 81.995])
    assert [pct(p, "2026-05-25", "tech") for p in prices] == pytest.approx([87.745, 87.745, 86.745, 85.495, 79.995])
    assert all(pct(p, "2026-11-01", "general") is None for p in prices)
    # 생략 시: 일반용역이 판정 불가라 최댓값을 알 수 없으므로 None
    assert all(pct(p, "2026-11-01", None) is None for p in prices)
    # 경계값: 2억은 2~5억 구간
    assert pct(200_000_000, "2026-11-01", "tech") == pytest.approx(88.745)


def test_resolve_lower_limit_local_service_and_unspecified_kind():
    """지자체 용역 하한율은 2026-10 공고 실측값(2억/5억/10억 경계, 조달청 상향 전 값과 같음). 일반용역 10억 이상은 판정 불가."""
    def pct(price, as_of, kind=None):
        r = R.resolve_lower_limit("서울특별시 강남구", work_type=2, est_price=price, as_of=as_of, servc_kind=kind)
        return None if r is None else r["lower_limit_pct"]

    prices = [50_000_000, 150_000_000, 300_000_000, 700_000_000, 2_000_000_000]
    assert [pct(p, "2026-11-01", "general") for p in prices] == pytest.approx([87.745, 87.745, 86.745, 85.495, None])
    assert [pct(p, "2026-11-01", "tech") for p in prices] == pytest.approx([87.745, 87.745, 86.745, 85.495, 79.995])
    # 종류 생략: 두 종류가 같으면 그 값, 일반이 판정 불가인 10억 이상은 None
    assert [pct(p, "2026-11-01") for p in prices] == pytest.approx([87.745, 87.745, 86.745, 85.495, None])
    # 상향 전 행
    assert pct(300_000_000, "2025-04-30") == pytest.approx(87.745)
    # 교육청도 같은 분류
    assert R.resolve_lower_limit("경기도교육청", work_type=2, est_price=300_000_000, as_of="2026-11-01")["lower_limit_pct"] == pytest.approx(86.745)
