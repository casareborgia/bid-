"""Phase 8-B: 계약 법령·규정 변경 감시(src/monitor_rules.py) 4종 종료 코드(0/3/4/6) 및 --accept 테스트."""
import datetime as dt
import json
from pathlib import Path

import monitor_rules as MR


def _sample_reg(tmp_path: Path) -> Path:
    reg = {
        "version": "2026-10-31",
        "items": [
            {
                "key": "yega_criteria",
                "kind": "admrul",
                "name": "(계약예규) 예정가격작성기준",
                "rule_id": "27944",
                "last_seen": {"serial": "100", "ef_date": "2026-08-25"},
                "model_keywords": ["복수예비가격", "기초금액", "15개"],
                "related_to": ["① 예가 범위"],
            },
            {
                "key": "koen_rule",
                "kind": "pi",
                "name": "한국남동발전 계약규정",
                "rule_id": "2200000160749",
                "last_seen": {"serial": "200", "ef_date": "2026-09-09"},
                "model_keywords": ["제34조", "±2%"],
                "related_to": ["① 예가 범위"],
            },
        ],
    }
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
    return p


def test_monitor_rules_exit_codes_and_accept_guard(tmp_path, capsys):
    reg_path = _sample_reg(tmp_path)
    out_md = tmp_path / "report.md"

    # 1. 변경 없음 -> 종료 코드 0
    def fetch_unchanged(item):
        return {
            "serial": item["last_seen"]["serial"],
            "ef_date": item["last_seen"]["ef_date"],
            "summary": "변경 없음",
        }

    code0 = MR.main(["--registry", str(reg_path), "--out", str(out_md)], fetcher=fetch_unchanged)
    assert code0 == 0 and out_md.is_file()
    assert "이상 없음" in out_md.read_text(encoding="utf-8")
    capsys.readouterr()

    # 2. 무관 변경(키워드 미적중) -> 종료 코드 6, --accept 없으면 레지스트리 불변
    def fetch_irrelevant(item):
        if item["key"] == "yega_criteria":
            return {"serial": "101", "ef_date": "2026-10-01", "summary": "용어 정비 및 단순 오기 정정"}
        return fetch_unchanged(item)

    code6 = MR.main(["--registry", str(reg_path), "--out", str(out_md)], fetcher=fetch_irrelevant)
    assert code6 == 6
    assert MR.load_registry(reg_path)["items"][0]["last_seen"]["serial"] == "100"
    capsys.readouterr()

    # 3. 모델 관련 변경(키워드 적중 + 시행예정 D-10 경보) -> 종료 코드 3
    def fetch_model_change(item):
        if item["key"] == "koen_rule":
            Tomorrow = (dt.date.today() + dt.timedelta(days=10)).isoformat()
            return {
                "serial": "201",
                "ef_date": "2026-10-15",
                "upcoming_ef_date": Tomorrow,
                "summary": "제34조 복수예비가격 범위 ±2% 조항 개정",
            }
        return fetch_unchanged(item)

    code3 = MR.main(["--registry", str(reg_path), "--out", str(out_md), "--json"], fetcher=fetch_model_change)
    assert code3 == 3
    js = json.loads(capsys.readouterr().out)
    assert js["exit_code"] == 3 and len(js["model_changes"]) == 1 and len(js["upcoming_alerts"]) == 1
    assert MR.load_registry(reg_path)["items"][1]["last_seen"]["serial"] == "200"

    # --accept 지정 시에만 레지스트리가 갱신됨
    code3_acc = MR.main(["--registry", str(reg_path), "--out", str(out_md), "--accept"], fetcher=fetch_model_change)
    assert code3_acc == 3
    assert MR.load_registry(reg_path)["items"][1]["last_seen"]["serial"] == "201"
    capsys.readouterr()

    # 4. API 오류 -> 종료 코드 4
    code4 = MR.main(
        ["--registry", str(reg_path), "--out", str(out_md)],
        fetcher=lambda item: {"error": "네트워크 연결 실패"},
    )
    assert code4 == 4
    capsys.readouterr()


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "law"
TODAY = dt.date(2026, 10, 31)


def _fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parsers_on_real_cli_output():
    """korean-law CLI 실출력 녹화본에서 식별자별 현행 일련번호·시행일을 뽑는다."""
    assert MR.parse_law_bulk(_fx("law_bulk.txt"), "002652", TODAY)["serial"] == "285893"
    loc = MR.parse_law_bulk(_fx("law_bulk.txt"), "010098", TODAY)
    assert loc["serial"] == "286149" and loc["ef_date"] == "2026-06-03" and loc["upcoming_ef_date"] is None
    assert MR.parse_law_bulk(_fx("law_bulk.txt"), "001552", TODAY) is None  # 없는 ID 는 None

    yc = MR.parse_admrul_history(_fx("admrul_history_27944.txt"), "27944", TODAY)
    assert yc == {"serial": "2100000284228", "ef_date": "2026-08-25", "upcoming_ef_date": None}
    lc = MR.parse_admrul_history(_fx("admrul_history_29508.txt"), "29508", TODAY)
    assert lc["serial"] == "2100000281382" and lc["ef_date"] == "2026-07-01"
    assert MR.parse_admrul_history(_fx("admrul_history_27944.txt"), "29508", TODAY) is None

    pi = MR.parse_pi_search(_fx("pi_search_koen.txt"), "(한국남동발전주식회사) 계약규정", TODAY)
    assert pi == {"serial": "2200000160749", "ef_date": "2026-09-09", "upcoming_ef_date": None}
    assert MR.parse_pi_search(_fx("pi_search_koen.txt"), "계약사무규정", TODAY) is None


def test_admrul_upcoming_version_detected():
    text = _fx("admrul_history_27944.txt").replace(
        "  시행 2026.08.25", "  시행 2026.11.20 | 발령 2026.10.30 제180호 일부개정 | 2100000290000\n  시행 2026.08.25", 1)
    yc = MR.parse_admrul_history(text, "27944", TODAY)
    assert yc["serial"] == "2100000284228" and yc["upcoming_ef_date"] == "2026-11-20"


def test_real_fetch_parse_failure_is_error_and_unknown_change_is_model_relevant(tmp_path, monkeypatch):
    reg = {"items": [{"key": "koen", "kind": "pi", "name": "한국남동발전 계약규정", "title": "(한국남동발전주식회사) 계약규정",
                      "last_seen": {"serial": "2200000150000", "ef_date": "2026-01-01"}, "model_keywords": ["제34조"]}]}
    # 응답 형식이 바뀌어 대상 행을 못 찾으면 '변경 없음'이 아니라 오류(4)
    monkeypatch.setattr(MR.LC, "public_institution_rules", lambda q: {"ok": True, "text": "형식이 바뀐 응답"})
    assert MR.evaluate_registry(reg, today=TODAY)["exit_code"] == 4
    # 실조회로 일련번호가 바뀌었는데 개정 내용을 모르면 무관(6)이 아니라 모델 관련(3)으로 분류
    monkeypatch.setattr(MR.LC, "public_institution_rules", lambda q: {"ok": True, "text": _fx("pi_search_koen.txt")})
    s = MR.evaluate_registry(reg, today=TODAY)
    assert s["exit_code"] == 3 and s["model_changes"][0]["current"]["serial"] == "2200000160749"


def test_registry_identifiers_match_recorded_cli_output():
    reg = MR.load_registry()
    by = {i["key"]: i for i in reg["items"]}
    for key, fx, fn, ident in [
        ("state_contract_decree", "law_bulk.txt", MR.parse_law_bulk, "law_id"),
        ("local_contract_decree", "law_bulk.txt", MR.parse_law_bulk, "law_id"),
        ("yega_criteria", "admrul_history_27944.txt", MR.parse_admrul_history, "rule_id"),
        ("local_exec_criteria", "admrul_history_29508.txt", MR.parse_admrul_history, "rule_id"),
        ("koen_contract_rule", "pi_search_koen.txt", MR.parse_pi_search, "title"),
    ]:
        got = fn(_fx(fx), by[key][ident], TODAY)
        assert got is not None, key
        assert {"serial": got["serial"], "ef_date": got["ef_date"]} == by[key]["last_seen"], key
