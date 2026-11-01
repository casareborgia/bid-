"""계약 법령·행정규칙·공공기관 규정 변경 감시 스크립트 (Phase 8-B).

레지스트리(`data/rules/registry.json`)에 등록된 법령·계약예규·기관 규정의 최신 일련번호(serial/MST)와 시행일(ef_date)을
조회해 `last_seen` 과 비교하고, 변경 건이 있으면 모델 관련 키워드 적중 여부로 "모델 관련 변경 / 무관 변경"을 분류한다.

종료 코드:
  0: 변경 없음
  3: 모델 관련 변경 감지 (섹션 8-F 조치 프로토콜에 따라 규칙표/모델 검토)
  4: API·네트워크 오류
  6: 무관 변경만 감지 (`--accept` 로 레지스트리 갱신 가능)

주의: `--accept` 옵션을 명시하지 않으면 `registry.json` 은 절대 수정되지 않는다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import law_client as LC  # noqa: E402

REGISTRY_PATH = ROOT / "data" / "rules" / "registry.json"
OUT_DIR = ROOT / "out"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(reg: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iso(d: str) -> str:
    """'2026.08.25' / '20260825' / '2026-08-25' → '2026-08-25'."""
    raw = re.sub(r"\D", "", d)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def _future(dates: list[str], today: dt.date) -> str | None:
    fut = sorted(x for x in dates if dt.date.fromisoformat(x) > today)
    return fut[0] if fut else None


def parse_law_bulk(text: str, law_id: str, today: dt.date) -> dict[str, Any] | None:
    """search_law_bulk 텍스트에서 법령ID 행의 MST·시행일(·시행예정일)을 찾는다.

    예: '   ID 002652 | MST 285893 | 시행 2026-06-03'
    """
    for line in text.splitlines():
        m = re.search(r"ID (\d+) \| MST (\d+) \| 시행 (\d{4}-\d{2}-\d{2})(.*)", line)
        if m and m.group(1) == str(law_id):
            ups = [_iso(x) for x in re.findall(r"(\d{4}[-.]\d{2}[-.]\d{2})", m.group(4))]
            return {"serial": m.group(2), "ef_date": m.group(3), "upcoming_ef_date": _future(ups, today)}
    return None


def parse_admrul_history(text: str, rule_id: str, today: dt.date) -> dict[str, Any] | None:
    """search_admin_rule(history=true) 텍스트에서 행정규칙ID 블록의 [현행] 일련번호·시행일을 찾는다.

    예: '■ (계약예규) 예정가격작성기준 (행정규칙ID 27944, 버전 27개)'
        '  시행 2026.08.25 | 발령 2026.08.25 제171호 일부개정 | 2100000284228 [현행]'
    """
    in_block = False
    current = None
    versions: list[str] = []
    for line in text.splitlines():
        if line.startswith("■"):
            in_block = f"행정규칙ID {rule_id}," in line or f"행정규칙ID {rule_id})" in line
            continue
        if not in_block:
            continue
        m = re.search(r"시행 (\d{4}\.\d{2}\.\d{2}) \|.*\| (\d{10,15})", line)
        if not m:
            continue
        ef = _iso(m.group(1))
        versions.append(ef)
        if "[현행]" in line and current is None:
            current = {"serial": m.group(2), "ef_date": ef}
    if current is None:
        return None
    return {**current, "upcoming_ef_date": _future(versions, today)}


def parse_pi_search(text: str, title: str, today: dt.date) -> dict[str, Any] | None:
    """search_public_institution_rules 텍스트에서 제목이 정확히 일치하는 규정의 일련번호·시행일을 찾는다.

    예: '[2200000160749] (한국남동발전주식회사) 계약규정' 다음 줄들에 '시행일: 20260909'
    """
    entries: list[tuple[str, str]] = []
    cur_serial = None
    for line in text.splitlines():
        m = re.match(r"\[(\d+)\] (.+)$", line.strip())
        if m:
            cur_serial = m.group(1) if m.group(2).strip() == title else None
            continue
        m = re.search(r"시행일: (\d{8})", line)
        if m and cur_serial:
            entries.append((_iso(m.group(1)), cur_serial))
            cur_serial = None
    if not entries:
        return None
    effective = [e for e in entries if dt.date.fromisoformat(e[0]) <= today] or entries
    ef, serial = max(effective)
    return {"serial": serial, "ef_date": ef, "upcoming_ef_date": _future([e[0] for e in entries], today)}


def fetch_item_status(item: dict[str, Any], fetcher=None, today: dt.date | None = None) -> dict[str, Any]:
    """단일 레지스트리 항목의 최신 상태를 조회한다.

    응답에서 대상 행을 찾지 못하면 이전 값으로 대체하지 않고 오류로 돌려준다(형식 변경을 '변경 없음'으로 오판하지 않도록).
    실조회 결과에는 개정 내용 요약이 없으므로 summary_known=False 로 표시한다.
    """
    if fetcher is not None:
        return fetcher(item)
    today = today or dt.date.today()
    kind = item.get("kind")
    if kind == "admrul":
        resp = LC.admin_rule_history(item.get("query") or item["name"])
        parse = lambda t: parse_admrul_history(t, str(item["rule_id"]), today)  # noqa: E731
    elif kind == "pi":
        resp = LC.public_institution_rules(item.get("query") or item["name"])
        parse = lambda t: parse_pi_search(t, item["title"], today)  # noqa: E731
    else:
        resp = LC.law_bulk([item["name"]])
        parse = lambda t: parse_law_bulk(t, str(item["law_id"]), today)  # noqa: E731
    if "error" in resp:
        return {"error": resp["error"]}
    info = parse(str(resp.get("text") or ""))
    if info is None:
        return {"error": "응답에서 대상 규정을 찾지 못했습니다(응답 형식 변경 또는 식별자 불일치) — 레지스트리 식별자와 CLI 출력을 확인하세요"}
    return {**info, "summary": "", "summary_known": False}


def evaluate_registry(
    reg: dict[str, Any],
    *,
    fetcher=None,
    today: dt.date | None = None,
) -> dict[str, Any]:
    """전체 레지스트리 항목을 점검하고 분류 결과를 반환한다."""
    today = today or dt.date.today()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    model_changes: list[dict[str, Any]] = []
    irrelevant_changes: list[dict[str, Any]] = []
    upcoming_alerts: list[dict[str, Any]] = []

    for item in reg.get("items", []):
        st = fetch_item_status(item, fetcher=fetcher, today=today)
        if "error" in st:
            errors.append(f"{item['name']}: {st['error']}")
            results.append({"key": item["key"], "name": item["name"], "status": "error", "error": st["error"]})
            continue

        last = item.get("last_seen") or {}
        cur_serial = str(st.get("serial") or "")
        cur_ef = str(st.get("ef_date") or "")
        changed = (cur_serial != str(last.get("serial") or "")) or (cur_ef != str(last.get("ef_date") or ""))

        summary = str(st.get("summary") or "")
        kws = item.get("model_keywords") or []
        hit_kws = [k for k in kws if k and k in summary]
        if changed and st.get("summary_known") is False:
            # 개정 내용을 모르면 무관 변경으로 넘기지 않고 사람이 검토하도록 모델 관련 변경으로 분류한다
            hit_kws = ["(개정 내용 미확인 — 수동 검토)"]

        # 시행예정(D-30 이내) 감지
        up_ef = st.get("upcoming_ef_date")
        if up_ef:
            try:
                d_up = dt.date.fromisoformat(str(up_ef))
                days_left = (d_up - today).days
                if 0 <= days_left <= 30:
                    upcoming_alerts.append({
                        "key": item["key"],
                        "name": item["name"],
                        "upcoming_ef_date": str(d_up),
                        "days_left": days_left,
                    })
            except Exception:
                pass

        entry = {
            "key": item["key"],
            "kind": item.get("kind"),
            "name": item["name"],
            "last_seen": last,
            "current": {"serial": cur_serial, "ef_date": cur_ef},
            "changed": changed,
            "hit_keywords": hit_kws,
            "model_relevant": bool(changed and hit_kws),
            "related_to": item.get("related_to") or [],
        }
        results.append(entry)
        if changed:
            if hit_kws:
                model_changes.append(entry)
            else:
                irrelevant_changes.append(entry)

    if errors:
        exit_code = 4
    elif model_changes:
        exit_code = 3
    elif irrelevant_changes:
        exit_code = 6
    else:
        exit_code = 0

    return {
        "checked_at": today.isoformat(),
        "exit_code": exit_code,
        "results": results,
        "model_changes": model_changes,
        "irrelevant_changes": irrelevant_changes,
        "upcoming_alerts": upcoming_alerts,
        "errors": errors,
    }


def write_rule_report(summary: dict[str, Any], path: Path) -> str:
    lines = [
        f"# 계약 법령·규정 변경 감시 리포트 — {summary['checked_at']}",
        "",
        f"- 종료 코드: **{summary['exit_code']}** (0=변경없음, 3=모델관련변경, 4=API오류, 6=무관변경만)",
        "",
    ]
    if summary.get("upcoming_alerts"):
        lines += ["## 시행예정(D-30 이내) 경보", ""]
        for u in summary["upcoming_alerts"]:
            lines.append(f"- **{u['name']}**: 시행예정일 `{u['upcoming_ef_date']}` (D-{u['days_left']})")
        lines.append("")

    lines += [
        "## 점검 결과 표",
        "",
        "| 구분 | 규정명 | 기존(serial / 시행일) | 현재(serial / 시행일) | 상태 | 적중 키워드 |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        if r.get("status") == "error":
            err_one = " ".join(str(r.get("error")).split()).replace("|", "/")
            lines.append(f"| - | {r['name']} | - | - | 오류 | {err_one} |")
            continue
        ls = r["last_seen"]
        cu = r["current"]
        st_txt = "변경 없음"
        if r["changed"]:
            st_txt = "**모델 관련 변경**" if r["model_relevant"] else "무관 변경"
        kws = ", ".join(r["hit_keywords"]) if r["hit_keywords"] else "-"
        lines.append(
            f"| {r['kind']} | {r['name']} | {ls.get('serial')} ({ls.get('ef_date')}) | "
            f"{cu.get('serial')} ({cu.get('ef_date')}) | {st_txt} | {kws} |"
        )

    lines += ["", "## 판정", ""]
    code = summary["exit_code"]
    if code == 0:
        lines.append("- **이상 없음**: 모든 감시 대상 법령·예규·공공기관 규정이 레지스트리 기준과 동일합니다.")
    elif code == 3:
        lines.append("- **모델 관련 규정 변경 감지 (종료 코드 3)**:")
        for m in summary["model_changes"]:
            lines.append(f"  - `{m['name']}`: {m['current']['serial']} ({m['current']['ef_date']}) — 키워드 {m['hit_keywords']} / 영향 {m['related_to']}")
        lines.append("- 조치: 작업계획서 섹션 8-F 조치 프로토콜에 따라 `data/rules/*.json` 규칙표를 검토·갱신한 뒤 `--accept` 로 레지스트리를 승인하세요.")
    elif code == 6:
        lines.append("- **모델 무관 변경만 감지 (종료 코드 6)**: 모델 관련 키워드 적중 없음. 내용 확인 후 `--accept` 로 레지스트리를 갱신하세요.")
    else:
        lines.append("- **조회 오류 (종료 코드 4)**:")
        for e in summary["errors"]:
            lines.append(f"  - {' '.join(str(e).split())}")

    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None, *, fetcher=None) -> int:
    ap = argparse.ArgumentParser(description="계약 법령·행정규칙·공공기관 규정 변경 감시")
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH, help="레지스트리 JSON 경로")
    ap.add_argument("--out", type=Path, default=None, help="리포트 출력 경로 (기본: out/rule_check_<YYYY-MM-DD>.md)")
    ap.add_argument("--accept", action="store_true", help="확인된 변경사항의 serial·ef_date 를 레지스트리에 반영")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    args = ap.parse_args(argv)

    reg = load_registry(args.registry)
    today = dt.date.today()
    summary = evaluate_registry(reg, fetcher=fetcher, today=today)
    out_path = args.out or (OUT_DIR / f"rule_check_{today.isoformat()}.md")
    report_txt = write_rule_report(summary, out_path)

    if args.accept and summary["exit_code"] in (0, 3, 6):
        cur_map = {r["key"]: r["current"] for r in summary["results"] if "current" in r}
        for item in reg.get("items", []):
            if item["key"] in cur_map:
                item["last_seen"] = cur_map[item["key"]]
        save_registry(reg, args.registry)
        summary["accepted"] = True
    else:
        summary["accepted"] = False

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(report_txt, end="")
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
