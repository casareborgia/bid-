"""법제처 국가법령정보 Open API / korean-law-mcp 연결 클라이언트 (Phase 8-A).

- 예측 코드(predict_cli, bid_tools, features, train_model)는 이 모듈을 호출하지 않는다.
- 호출은 규정 변경 감시(monitor_rules.py)에서만 사용한다.
- 1순위: 로컬 CLI (`korean-law <tool> --json-input <json>` subprocess, LAW_OC 있을 때). 출력은 JSON 이 아닌 텍스트다
- 2순위: 공개 MCP HTTP JSON-RPC 서버 (`https://mcp.gomdori.app/law`, 무상태 폴백)
- `[NOT_FOUND]` 마커나 `isError=True`, 타임아웃, 파싱 실패는 예외 대신 `{"error": ...}` 로 통일한다.
- 응답은 `out/law_cache/<sha256>.json` 에 캐시(기본 TTL 1일)하며, 어떤 경로에서도 `LAW_OC` 값을 출력하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "out" / "law_cache"
PUBLIC_MCP_URL = "https://mcp.gomdori.app/law"
DEFAULT_TTL = 86400  # 1일 (초)
DEFAULT_TIMEOUT = 30


def _redact(msg: str, secret: str | None) -> str:
    if not secret or len(secret) < 2:
        return msg
    return msg.replace(secret, "***")


def _cache_key(tool: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _read_cache(cache_dir: Path, key: str, ttl: int) -> dict[str, Any] | None:
    p = cache_dir / f"{key}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = float(data.get("_cached_at", 0))
        if ttl > 0 and (time.time() - ts) <= ttl and "response" in data:
            return data["response"]
    except Exception:
        return None
    return None


def _write_cache(cache_dir: Path, key: str, response: dict[str, Any]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{key}.json"
        p.write_text(
            json.dumps({"_cached_at": time.time(), "response": response}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _normalize_mcp_result(raw: dict[str, Any], secret: str | None = None) -> dict[str, Any]:
    """MCP tool call 응답(content 배열 또는 JSON dict)을 정규화한다."""
    if not isinstance(raw, dict):
        return {"error": "비정상 응답 형식"}
    if "error" in raw and isinstance(raw["error"], (str, dict)):
        err = raw["error"] if isinstance(raw["error"], str) else json.dumps(raw["error"], ensure_ascii=False)
        return {"error": _redact(err, secret)}
    res = raw.get("result", raw)
    if not isinstance(res, dict):
        return {"error": "비정상 result 형식"}
    is_err = bool(res.get("isError"))
    texts: list[str] = []
    for item in res.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    joined = "\n".join(texts).strip() if texts else str(res.get("text", "")).strip()
    if is_err or "[NOT_FOUND]" in joined:
        return {"error": _redact(joined or "조회 결과 없음 ([NOT_FOUND])", secret), "not_found": "[NOT_FOUND]" in joined}
    # 텍스트가 JSON 이면 파싱한 데이터도 함께 제공한다
    parsed = None
    if joined.startswith("{") or joined.startswith("["):
        try:
            parsed = json.loads(joined)
        except Exception:
            parsed = None
    return {"ok": True, "text": _redact(joined, secret), "data": parsed, "raw": res}


def _call_cli(tool: str, params: dict[str, Any], law_oc: str, timeout: int) -> dict[str, Any]:
    cli_bin = shutil.which("korean-law")
    if not cli_bin:
        raise FileNotFoundError("korean-law CLI 미설치")
    env = {**os.environ, "LAW_OC": law_oc}
    cmd = [cli_bin, tool, "--json-input", json.dumps(params, ensure_ascii=False)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, check=False)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or f"CLI 종료 코드 {proc.returncode}").strip()
        return {"error": _redact(err_msg, law_oc), "not_found": "[NOT_FOUND]" in err_msg}
    # korean-law CLI 는 사람이 읽는 텍스트를 출력한다(4.15 실측). JSON 이면 MCP 응답으로 정규화하고, 아니면 텍스트 그대로 쓴다.
    if out.startswith("{"):
        try:
            return _normalize_mcp_result(json.loads(out), secret=law_oc)
        except json.JSONDecodeError:
            pass
    if not out:
        return {"error": "CLI 응답이 비어 있습니다"}
    if "[NOT_FOUND]" in out:
        return {"error": _redact(out, law_oc), "not_found": True}
    return {"ok": True, "text": _redact(out, law_oc), "data": None}


def _call_http(tool: str, params: dict[str, Any], url: str, law_oc: str | None, timeout: int) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": params},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return _normalize_mcp_result(data, secret=law_oc)


def call(
    tool: str,
    params: dict[str, Any],
    *,
    cache_dir: Path = CACHE_DIR,
    ttl: int = DEFAULT_TTL,
    timeout: int = DEFAULT_TIMEOUT,
    prefer_cli: bool = True,
    http_url: str = PUBLIC_MCP_URL,
) -> dict[str, Any]:
    """korean-law-mcp 도구를 호출하고 정규화된 dict 를 반환한다.

    오류 시 예외를 던지지 않고 `{"error": ...}` 를 반환한다.
    """
    load_dotenv(ROOT / ".env")
    law_oc = os.environ.get("LAW_OC", "").strip() or None
    ckey = _cache_key(tool, params)
    if ttl > 0:
        cached = _read_cache(cache_dir, ckey, ttl)
        if cached is not None:
            return {**cached, "cached": True}

    result: dict[str, Any] | None = None
    if prefer_cli and law_oc:
        try:
            result = _call_cli(tool, params, law_oc, timeout)
        except FileNotFoundError:
            result = None  # CLI 가 없으면 HTTP 폴백
        except subprocess.TimeoutExpired:
            return {"error": f"CLI 호출 타임아웃 ({timeout}s)"}
        except Exception as e:
            return {"error": _redact(f"CLI 호출 오류: {e}", law_oc)}

    if result is None:
        try:
            result = _call_http(tool, params, http_url, law_oc, timeout)
        except requests.Timeout:
            return {"error": f"HTTP 호출 타임아웃 ({timeout}s)"}
        except ValueError as e:
            return {"error": f"JSON 파싱 실패: {_redact(str(e), law_oc)}"}
        except Exception as e:
            return {"error": _redact(f"HTTP 호출 오류: {e}", law_oc)}

    if result.get("ok") and ttl > 0:
        _write_cache(cache_dir, ckey, result)
    return {**result, "cached": False}


# ---------------------------------------------------------------- 고수준 래퍼 (Phase 8-A-4)
def admin_rule_history(query: str, **kw: Any) -> dict[str, Any]:
    """행정규칙 발령 연혁(행정규칙ID별 시행일·일련번호, [현행] 표시) 조회. query 는 행정규칙명 검색어."""
    return call("search_admin_rule", {"query": str(query), "history": True}, **kw)


def admin_rule_text(serial: str, jo: str | None = None, keyword: str | None = None, **kw: Any) -> dict[str, Any]:
    """행정규칙 일련번호(13자리)로 본문(또는 특정 조문/키워드) 조회. 행정규칙ID(4~5자리)로는 조회되지 않는다."""
    params: dict[str, Any] = {"id": str(serial)}
    if jo:
        params["jo"] = jo
    if keyword:
        params["keyword"] = keyword
    return call("get_admin_rule", params, **kw)


def public_institution_rules(query: str, **kw: Any) -> dict[str, Any]:
    """공공기관 규정(예: 한국남동발전 계약규정) 검색 — 일련번호·기관·공포일·시행일 목록."""
    return call("search_public_institution_rules", {"query": str(query)}, **kw)


def public_institution_rule_text(serial: str, **kw: Any) -> dict[str, Any]:
    """공공기관 규정 일련번호로 본문 조회."""
    return call("get_public_institution_rule_text", {"id": str(serial)}, **kw)


def law_bulk(names: list[str], previous: dict[str, str] | None = None, **kw: Any) -> dict[str, Any]:
    """다수 법령의 최신 법령ID·MST·시행일·시행예정 일괄 조회."""
    params: dict[str, Any] = {"queries": list(names)}
    if previous:
        params["previous"] = previous
    return call("search_law_bulk", params, **kw)


def verify_citations(text: str, **kw: Any) -> dict[str, Any]:
    """텍스트 내 법령 조문 인용 실존 검증."""
    return call("verify_citations", {"text": text}, **kw)
