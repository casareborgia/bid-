"""공고번호 조회(bid_notice) 테스트. 네트워크 없이 실응답 녹화본(tests/fixtures/g2b/, 2026-11-01, 담당자 정보 마스킹)을 쓴다."""
import copy
import json
from pathlib import Path

import pytest

import bid_notice as BN
from collect_g2b_api import parse_body

FIX = Path(__file__).parent / "fixtures" / "g2b"


def _fx(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text())


def _items(name: str) -> list[dict]:
    return _fx(name)["response"]["body"]["items"]


class FakeClient:
    """op → 녹화본 이름. 호출 순서를 기록한다."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.calls: list[tuple[str, dict]] = []

    def get(self, op: str, params: dict) -> dict:
        self.calls.append((op, params))
        return parse_body(_fx(self.mapping.get(op, "notice_cnstwk_empty")))


CNSTWK = {"getBidPblancListInfoCnstwk": "notice_cnstwk_R26BK01752065",
          "getBidPblancListInfoCnstwkBsisAmount": "bsis_cnstwk_R26BK01752065"}
SERVC = {"getBidPblancListInfoServc": "notice_servc_R26BK01749660",
         "getBidPblancListInfoServcBsisAmount": "bsis_servc_R26BK01749660"}


def test_fetch_cnstwk_notice():
    c = FakeClient(CNSTWK)
    n = BN.fetch_notice(" R26BK01752065 ", client=c)
    assert [op for op, _ in c.calls] == ["getBidPblancListInfoCnstwk", "getBidPblancListInfoCnstwkBsisAmount"]
    assert c.calls[0][1]["inqryDiv"] == "2" and c.calls[0][1]["bidNtceNo"] == "R26BK01752065"
    assert n["work_type"] == 1 and n["agency"] == "서울특별시" and n["notice_date"] == "2026-10-01"
    assert n["est_price"] == 167_746_809 and n["base_amt"] == 184_521_490
    assert n["range_pct"] == 3 and n["lower_limit_pct"] == 89.745
    assert n["a_value"] == 11_684_881
    assert set(n["a_items"]) == {"국민연금보험료", "국민건강보험료", "노인장기요양보험료", "퇴직공제부금", "산업안전보건관리비"}
    assert n["servc_kind"] is None and n["warnings"] == []


def test_a_value_matches_a_info_operation():
    """기초금액 응답의 A값 항목은 입찰가격산식A정보조회 응답과 같은 값이다(녹화본 대조)."""
    bsis_total, _ = BN.a_value_from(_items("bsis_cnstwk_R26BK01752065")[0])
    a_total, _ = BN.a_value_from(_items("a_value_R26BK01752065")[0])
    assert bsis_total == a_total == 11_684_881


def test_fetch_servc_falls_back_from_cnstwk():
    c = FakeClient(SERVC)
    n = BN.fetch_notice("R26BK01749660", client=c)
    assert [op for op, _ in c.calls] == ["getBidPblancListInfoCnstwk", "getBidPblancListInfoServc",
                                         "getBidPblancListInfoServcBsisAmount"]
    assert n["work_type"] == 2 and n["servc_kind"] == "tech"
    assert n["est_price"] == 62_200_000 and n["base_amt"] == 68_420_000 and n["range_pct"] == 3
    assert n["lower_limit_pct"] == 88 and n["a_value"] == 0 and n["a_items"] == {}


def test_not_found_raises_lookup_error():
    with pytest.raises(LookupError, match="찾지 못했습니다"):
        BN.fetch_notice("NOPE", client=FakeClient({}))


def test_missing_bsis_amount_warns():
    c = FakeClient({"getBidPblancListInfoCnstwk": "notice_cnstwk_R26BK01752065"})
    n = BN.fetch_notice("R26BK01752065", client=c)
    assert n["base_amt"] is None and n["range_pct"] is None and n["a_value"] == 0
    assert any("기초금액 미공개" in w for w in n["warnings"])
    assert n["lower_limit_pct"] == 89.745          # 하한율은 공고 목록에 있다


def test_a_value_flags():
    item = copy.deepcopy(_items("bsis_cnstwk_R26BK01752065")[0])
    item["qltyMngcst"] = "500000"
    assert BN.a_value_from(item)[0] == 11_684_881 + 500_000          # qltyMngcstAObjYn=Y
    item["qltyMngcstAObjYn"] = "N"
    assert BN.a_value_from(item)[0] == 11_684_881
    item["bidPrceCalclAYn"] = "N"
    assert BN.a_value_from(item) == (0.0, {})


def test_unusual_range_and_single_price_warn():
    notice = copy.deepcopy(_items("notice_cnstwk_R26BK01752065")[0])
    bsis = copy.deepcopy(_items("bsis_cnstwk_R26BK01752065")[0])
    bsis["rsrvtnPrceRngBgnRate"], bsis["rsrvtnPrceRngEndRate"] = "-2.5", "+2.5"
    notice["prearngPrceDcsnMthdNm"] = "단일예가"
    n = BN.parse_notice(notice, bsis, 1)
    assert n["range_pct"] is None
    assert any("±2/±3" in w for w in n["warnings"]) and any("단일예가" in w for w in n["warnings"])


def test_make_client_uses_bid_public_info_service(monkeypatch):
    monkeypatch.setattr(BN, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("G2B_SERVICE_KEY", "dummy-key")
    assert BN.make_client().base_url == BN.BASE_URL
    monkeypatch.delenv("G2B_SERVICE_KEY")
    with pytest.raises(ValueError, match="G2B_SERVICE_KEY"):
        BN.make_client()


def test_fixtures_have_no_personal_info():
    for p in FIX.glob("*.json"):
        text = p.read_text()
        assert "@" not in text and "serviceKey" not in text.lower().replace("servicekey_", "")
        for it in json.loads(text)["response"]["body"]["items"]:
            for f in ("ntceInsttOfclNm", "ntceInsttOfclTelNo", "ntceInsttOfclEmailAdrs", "exctvNm"):
                assert it.get(f, "***") == "***", (p.name, f)
