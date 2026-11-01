"""발주기관명 기반 예가 범위(±2%/±3%) 및 적격심사 낙찰하한율 규칙 엔진 (Phase 8-C, 8-D).

- 예측 코드(predict_cli, bid_tools)는 법제처 API 를 직접 호출하지 않고 이 모듈을 통해 로컬 규칙표 JSON 만 읽는다.
- 예가 범위 자동 판정은 status="verified" 행만 사용하며, 미확인("unverified") 행은 판정에서 제외한다.
- 실데이터(data/g2b_detail_all.parquet)의 수요기관(dminsttNm)별 실제 range_cls 분포와 교차검증해 일치율을 리포트한다.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REGIME_RULES_PATH = ROOT / "data" / "rules" / "regime_rules.json"
LOWER_LIMIT_RULES_PATH = ROOT / "data" / "rules" / "lower_limit_rules.json"
G2B_PATH = ROOT / "data" / "g2b_detail_all.parquet"
OUT_CHECK_PATH = ROOT / "out" / "regime_rules_check.md"


def load_regime_rules(path: Path = REGIME_RULES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_lower_limit_rules(path: Path = LOWER_LIMIT_RULES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date_in_range(as_of: str | dt.date | pd.Timestamp | None, valid_from: str | None, valid_to: str | None) -> bool:
    if as_of is None:
        d = dt.date.today()
    elif isinstance(as_of, pd.Timestamp):
        d = as_of.date()
    elif isinstance(as_of, dt.datetime):
        d = as_of.date()
    elif isinstance(as_of, dt.date):
        d = as_of
    else:
        d = pd.Timestamp(str(as_of)).date()
    if valid_from and d < pd.Timestamp(valid_from).date():
        return False
    if valid_to and d > pd.Timestamp(valid_to).date():
        return False
    return True


def _match_scope(agency: str, scope: dict[str, Any]) -> bool:
    name = agency.strip()
    if not name:
        return False
    for ex in scope.get("exclude_keywords") or []:
        if ex and ex in name:
            return False
    inc_kw = scope.get("include_keywords") or []
    inc_rx = scope.get("include_regex")
    matched = False
    if any(k in name for k in inc_kw if k):
        matched = True
    if not matched and inc_rx:
        if re.search(inc_rx, name):
            matched = True
    return matched


def format_basis(basis: dict[str, Any] | None) -> str:
    if not basis:
        return ""
    parts = [basis.get("rule") or ""]
    if basis.get("article"):
        parts.append(str(basis["article"]))
    meta = []
    if basis.get("serial"):
        meta.append(f"일련번호 {basis['serial']}")
    if basis.get("ef_date"):
        meta.append(f"{basis['ef_date']} 시행")
    base_str = " ".join(p for p in parts if p).strip()
    if meta:
        base_str = f"{base_str} ({', '.join(meta)})"
    if basis.get("note"):
        base_str = f"{base_str} — {basis['note']}"
    return base_str


def resolve_range(
    agency: str | None,
    work_type: int = 2,
    as_of: str | dt.date | pd.Timestamp | None = None,
    rules_path: Path = REGIME_RULES_PATH,
    include_unverified: bool = False,
) -> dict[str, Any] | None:
    """발주기관명·공종·기준일로 예가 범위(±N%) 규칙을 판정한다.

    verified 행만 자동 판정에 쓰며, 매칭되는 행이 없거나 unverified 면 None 을 돌려준다.
    """
    if not agency or not str(agency).strip():
        return None
    data = load_regime_rules(rules_path)
    wt = int(work_type)
    for rule in data.get("rules", []):
        if not include_unverified and rule.get("status") != "verified":
            continue
        if wt not in (rule.get("work_type") or [1, 2]):
            continue
        if not _date_in_range(as_of, rule.get("valid_from"), rule.get("valid_to")):
            continue
        if _match_scope(str(agency), rule.get("scope") or {}):
            pct = int(rule["range_pct"])
            basis = rule.get("basis") or {}
            return {
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name"),
                "status": rule.get("status"),
                "range_pct": pct,
                "range_cls": f"±{pct}",
                "n_prices": int(rule.get("n_prices", 15)),
                "n_draw": int(rule.get("n_draw", 4)),
                "basis": basis,
                "basis_text": format_basis(basis),
            }
    return None


def resolve_lower_limit(
    agency: str | None,
    work_type: int = 2,
    est_price: float | None = None,
    as_of: str | dt.date | pd.Timestamp | None = None,
    rules_path: Path = LOWER_LIMIT_RULES_PATH,
    servc_kind: str | None = None,
) -> dict[str, Any] | None:
    """발주기관명·공종·추정가격·기준일로 적격심사 통상 낙찰하한율을 판정한다.

    용역 행에 servc_kind(general=일반용역, tech=기술용역)가 있으면 그 종류만 맞춘다. 생략하면 해당 구간의 두 종류 중
    하한율이 높은 행을 쓴다(하한율을 높게 잡을수록 투찰가가 올라가 하한 미달 위험이 줄어 보수적).
    단, 그 구간에 판정 불가(unverified)인 용역 종류 행이 있으면 최댓값을 알 수 없으므로 None 을 돌려준다.
    """
    if not agency or not str(agency).strip():
        return None
    data = load_lower_limit_rules(rules_path)
    wt = int(work_type)
    price = float(est_price) if est_price is not None else 0.0
    regime_id: str | None = None  # 지연 계산
    kind_candidates: list[dict[str, Any]] = []  # servc_kind 생략 시 종류별 후보
    unknown_kind = False  # servc_kind 생략 시 같은 구간에 판정 불가 종류 행이 있었는지
    for rule in data.get("rules", []):
        if rule.get("status") not in ("verified", "unverified"):
            continue
        if wt not in (rule.get("work_type") or [1, 2]):
            continue
        if not _date_in_range(as_of, rule.get("valid_from"), rule.get("valid_to")):
            continue
        if rule.get("servc_kind") and servc_kind and rule["servc_kind"] != servc_kind:
            continue
        p_min = rule.get("est_price_min")
        p_max = rule.get("est_price_max")
        if p_min is not None and price < float(p_min):
            continue
        if p_max is not None and price >= float(p_max):
            continue
        scope = rule.get("scope") or {}
        if scope.get("regime_rule_ids"):
            # 기관 분류를 예가 범위 규칙표의 판정에 위임한다 (verified 행만)
            if regime_id is None:
                rr = resolve_range(agency, work_type=wt, as_of=as_of)
                regime_id = rr["rule_id"] if rr else ""
            matched = regime_id in scope["regime_rule_ids"]
        else:
            matched = _match_scope(str(agency), scope)
        if matched and rule.get("status") != "verified":
            if rule.get("servc_kind") and not servc_kind:
                unknown_kind = True
            continue
        if matched:
            basis = rule.get("basis") or {}
            hit = {
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name"),
                "lower_limit_pct": float(rule["lower_limit_pct"]),
                "servc_kind": rule.get("servc_kind"),
                "basis": basis,
                "basis_text": format_basis(basis),
            }
            if rule.get("servc_kind") and not servc_kind:
                kind_candidates.append(hit)
                continue
            if kind_candidates:
                break  # 종류별 후보를 모으는 중에 다른 기관 행을 만나면 그 후보로 끝낸다
            return hit
    if kind_candidates and unknown_kind:
        return None
    if kind_candidates:
        best = max(kind_candidates, key=lambda h: h["lower_limit_pct"])
        if len({h["lower_limit_pct"] for h in kind_candidates}) > 1:
            best = {**best, "basis_text": best["basis_text"] + " — 용역 종류 미지정: 일반·기술 중 높은 하한율"}
        return best
    return None


def check_regime_rules_against_data(
    parquet_path: Path = G2B_PATH,
    rules_path: Path = REGIME_RULES_PATH,
    out_path: Path | None = OUT_CHECK_PATH,
) -> pd.DataFrame:
    """수집된 나라장터 parquet 의 dminsttNm 에 규칙표를 순서대로 적용해 실제 range_cls 일치율을 검증한다."""
    g = pd.read_parquet(parquet_path, columns=["dminsttNm", "work_type", "range_cls", "yega_rate"])
    g = g[g["yega_rate"].between(96.5, 103.5) & g["range_cls"].isin(["±2", "±3"]) & g["dminsttNm"].notna()].copy()

    rules_data = load_regime_rules(rules_path)
    rules = rules_data.get("rules", [])

    # 고유 기관명 단위로 먼저 매칭해 속도를 최적화한다
    uniq_agencies = g["dminsttNm"].astype(str).unique()
    agency_to_rule: dict[str, str] = {}
    for ag in uniq_agencies:
        for r in rules:
            if _match_scope(ag, r.get("scope") or {}):
                agency_to_rule[ag] = r["id"]
                break

    g["rule_id"] = g["dminsttNm"].astype(str).map(agency_to_rule).fillna("unmatched")

    rows = []
    for r in rules:
        rid = r["id"]
        sub = g[g["rule_id"] == rid]
        n = len(sub)
        r2 = int((sub["range_cls"] == "±2").sum())
        r3 = int((sub["range_cls"] == "±3").sum())
        expected_cls = f"±{int(r['range_pct'])}"
        matched = r2 if expected_cls == "±2" else r3
        match_rate = float(matched / n) if n > 0 else np.nan
        rows.append({
            "rule_id": rid,
            "name": r.get("name"),
            "status": r.get("status"),
            "expected_range": expected_cls,
            "n": n,
            "n_r2": r2,
            "n_r3": r3,
            "match_rate": match_rate,
        })

    unm = g[g["rule_id"] == "unmatched"]
    rows.append({
        "rule_id": "unmatched",
        "name": "기타/미분류 기관",
        "status": "unmatched",
        "expected_range": "-",
        "n": int(len(unm)),
        "n_r2": int((unm["range_cls"] == "±2").sum()),
        "n_r3": int((unm["range_cls"] == "±3").sum()),
        "match_rate": np.nan,
    })

    df = pd.DataFrame(rows)
    if out_path is not None:
        lines = [
            "# 예가 범위 규칙표(`regime_rules.json`) 실데이터 교차검증 리포트 (Phase 8-C)",
            "",
            f"- 생성: {dt.datetime.now():%Y-%m-%d %H:%M} | 대상: `{parquet_path.name}` 정상 범위(±2/±3) {len(g):,}건",
            "- 기준: `status == 'verified'` 규칙은 실데이터 일치율(`match_rate`) **95.0% 이상**이어야 자동 판정에 사용한다.",
            "",
            "| 규칙 ID | 기관 분류명 | 상태 | 규칙 범위 | 표본 수(n) | ±2% 건수 | ±3% 건수 | 규칙 일치율 | 판정 |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
        for _, row in df.iterrows():
            mr = "-" if pd.isna(row["match_rate"]) else f"{row['match_rate']*100:.2f}%"
            if row["status"] == "verified":
                verdict = "통과 (≥95%)" if (pd.notna(row["match_rate"]) and row["match_rate"] >= 0.95) else "미달 (<95%)"
            elif row["status"] == "unverified":
                verdict = "자동판정 제외 (혼재)"
            else:
                verdict = "수동 입력 대상"
            lines.append(
                f"| `{row['rule_id']}` | {row['name']} | {row['status']} | {row['expected_range']} | "
                f"{int(row['n']):,} | {int(row['n_r2']):,} | {int(row['n_r3']):,} | {mr} | {verdict} |"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df
