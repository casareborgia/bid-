"""Phase 8-A: 법제처 Open API 클라이언트(src/law_client.py) 오프라인 픽스처 테스트.

픽스처는 korean-law CLI 4.15.9 실출력 녹화본(2026-10-31, 인증키 마스킹)이다. CLI 는 JSON 이 아닌 텍스트를 출력한다.
"""
import json
from pathlib import Path
import subprocess
import pytest
import requests

import law_client as LC

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "law"


def _mcp(name: str, is_error: bool = False) -> dict:
    """녹화된 텍스트를 MCP tools/call 응답 형태로 감싼다 (HTTP 폴백 경로용)."""
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return {"result": {"content": [{"type": "text", "text": text}], "isError": is_error}}


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_call_http_success_and_cache_hit(tmp_path, monkeypatch):
    monkeypatch.delenv("LAW_OC", raising=False)
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)
    fixture_data = _mcp("admrul_history_27944.txt")
    calls = {"n": 0}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fixture_data

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return FakeResp()

    monkeypatch.setattr(LC.requests, "post", fake_post)
    r1 = LC.admin_rule_history("예정가격작성기준", cache_dir=tmp_path, ttl=3600)
    assert r1["ok"] is True and r1["cached"] is False and "2100000284228 [현행]" in r1["text"]
    assert calls["n"] == 1

    # 두 번째 호출은 캐시 적중(네트워크 미호출)
    r2 = LC.admin_rule_history("예정가격작성기준", cache_dir=tmp_path, ttl=3600)
    assert r2["ok"] is True and r2["cached"] is True
    assert calls["n"] == 1


def test_call_not_found_returns_error_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("LAW_OC", raising=False)
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)
    not_found_data = _mcp("not_found.txt", is_error=True)

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return not_found_data

    monkeypatch.setattr(LC.requests, "post", lambda *a, **k: FakeResp())
    r = LC.admin_rule_text("999999", cache_dir=tmp_path, ttl=0)
    assert "error" in r and r.get("not_found") is True


def test_call_cli_fallback_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("LAW_OC", "SECRET_OC_KEY_123")
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(LC.shutil, "which", lambda _: None)
    pi_data = _mcp("pi_search_koen.txt")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return pi_data

    monkeypatch.setattr(LC.requests, "post", lambda *a, **k: FakeResp())
    r = LC.public_institution_rules("한국남동발전 계약규정", cache_dir=tmp_path, ttl=0)
    assert r["ok"] is True and "[2200000160749] (한국남동발전주식회사) 계약규정" in r["text"]


def test_call_timeout_and_secret_redaction(tmp_path, monkeypatch):
    secret = "MY_SUPER_SECRET_OC"
    monkeypatch.setenv("LAW_OC", secret)
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(LC.shutil, "which", lambda _: "/usr/local/bin/korean-law")

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="korean-law", timeout=5)

    monkeypatch.setattr(LC.subprocess, "run", fake_run)
    r_to = LC.call("search_law", {"query": "국가계약법"}, cache_dir=tmp_path, ttl=0, timeout=5)
    assert "error" in r_to and "타임아웃" in r_to["error"]

    # CLI 오류 메시지에 인증키가 노출되어도 마스킹되어야 함
    class ErrProc:
        returncode = 1
        stdout = ""
        stderr = f"API failed with key={secret}"

    monkeypatch.setattr(LC.subprocess, "run", lambda *a, **k: ErrProc())
    r_err = LC.call("search_law", {"query": "국가계약법"}, cache_dir=tmp_path, ttl=0)
    assert "error" in r_err and secret not in r_err["error"] and "***" in r_err["error"]


def test_call_http_invalid_json_and_high_level_wrappers(tmp_path, monkeypatch):
    monkeypatch.delenv("LAW_OC", raising=False)
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)

    class BadJsonResp:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("Invalid JSON response")

    monkeypatch.setattr(LC.requests, "post", lambda *a, **k: BadJsonResp())
    r = LC.law_bulk(["국가계약법"], cache_dir=tmp_path, ttl=0)
    assert "error" in r and "JSON 파싱 실패" in r["error"]

    monkeypatch.setattr(LC.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("timeout")))
    r2 = LC.verify_citations("국가계약법 제7조", cache_dir=tmp_path, ttl=0)
    assert "error" in r2 and "타임아웃" in r2["error"]


def test_call_cli_text_output_and_real_param_names(tmp_path, monkeypatch):
    """CLI 는 텍스트를 출력한다: JSON 파싱 없이 ok 로 돌려주고, 도구별 실제 파라미터 이름을 넘긴다."""
    monkeypatch.setenv("LAW_OC", "SECRET_OC_KEY_123")
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(LC.shutil, "which", lambda _: "/opt/homebrew/bin/korean-law")
    seen = []

    def fake_run(cmd, **k):
        seen.append((cmd[1], json.loads(cmd[3])))
        return _Proc(stdout=(FIXTURES / "law_bulk.txt").read_text(encoding="utf-8"))

    monkeypatch.setattr(LC.subprocess, "run", fake_run)
    r = LC.law_bulk(["국가를 당사자로 하는 계약에 관한 법률 시행령"], cache_dir=tmp_path, ttl=0)
    assert r["ok"] is True and "MST 285893" in r["text"]
    LC.admin_rule_history("예정가격작성기준", cache_dir=tmp_path, ttl=0)
    LC.public_institution_rules("한국남동발전 계약규정", cache_dir=tmp_path, ttl=0)
    LC.admin_rule_text("2100000284228", keyword="복수예비가격", cache_dir=tmp_path, ttl=0)
    assert seen == [
        ("search_law_bulk", {"queries": ["국가를 당사자로 하는 계약에 관한 법률 시행령"]}),
        ("search_admin_rule", {"query": "예정가격작성기준", "history": True}),
        ("search_public_institution_rules", {"query": "한국남동발전 계약규정"}),
        ("get_admin_rule", {"id": "2100000284228", "keyword": "복수예비가격"}),
    ]


def test_call_cli_not_found_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("LAW_OC", "SECRET_OC_KEY_123")
    monkeypatch.setattr(LC, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(LC.shutil, "which", lambda _: "/opt/homebrew/bin/korean-law")
    nf = (FIXTURES / "not_found.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(LC.subprocess, "run", lambda *a, **k: _Proc(stderr=nf, returncode=1))
    r = LC.admin_rule_text("27944", cache_dir=tmp_path, ttl=0)
    assert "error" in r and r["not_found"] is True


def test_fixtures_have_no_secret():
    for p in FIXTURES.glob("*.txt"):
        t = p.read_text(encoding="utf-8")
        assert "OC=" not in t or "OC=***" in t, p.name
