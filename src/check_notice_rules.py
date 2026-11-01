"""낙찰하한율·예가 범위 규칙표를 나라장터 입찰공고로 대량 검증한다.

입찰공고 목록의 낙찰하한율(sucsfbidLwltRate)과 기초금액 조회의 예가 범위(rsrvtnPrceRng*)는 발주기관이 공고에 적은 값이다.
규칙표(`rules.resolve_lower_limit`, `rules.resolve_range`)가 같은 공고에 대해 내는 판정과 비교해 일치율과 불일치 유형을 낸다.
하한율 규칙표는 그동안 실데이터 검증이 없었다(예가 범위는 538,907건 개찰 데이터로 검증, out/regime_rules_check.md).

호출: 공고 목록·기초금액 × 공사·용역, 월 1개월 약 40~50페이지(999건/페이지). 원시 응답은 out/notice_raw/ 에 저장해 재사용한다.

사용 예
  uv run python src/check_notice_rules.py --month 2026-10            # 수집(캐시 재사용) + 리포트 out/notice_rules_check_2026-10.md
  uv run python src/check_notice_rules.py --month 2026-10 --no-fetch # 캐시만으로 리포트
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bid_notice as BN  # noqa: E402
import collect_g2b_api as cg  # noqa: E402
import rules as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "out" / "notice_raw"
OUT_DIR = ROOT / "out"
OPS = {(1, "notice"): BN.NOTICE_OP[1], (2, "notice"): BN.NOTICE_OP[2],
       (1, "bsis"): BN.BSIS_OP[1], (2, "bsis"): BN.BSIS_OP[2]}
KEEP = ["bidNtceNo", "bidNtceOrd", "bidNtceNm", "bidNtceDt", "dminsttNm", "ntceInsttNm", "presmptPrce", "sucsfbidLwltRate",
        "sucsfbidMthdNm", "sucsfbidMthdAppStd", "cntrctCnclsMthdNm", "prearngPrceDcsnMthdNm", "srvceDivNm",
        "bssamt", "rsrvtnPrceRngBgnRate", "rsrvtnPrceRngEndRate", "bidClsfcNo"]


def method_category(name: str | None) -> str:
    """낙찰방법명 → 비교 그룹. 규칙표는 적격심사 통상 하한율만 담는다."""
    n = str(name or "")
    for key, cat in (("소액수의", "소액수의견적"), ("적격심사", "적격심사"), ("제한적최저가", "제한적최저가"),
                     ("최저가", "최저가"), ("협상", "협상"), ("종합심사", "종합심사")):
        if key in n:
            return cat
    return "기타"


def fetch_month(ym: str, client=None, cache_dir: Path = RAW_DIR, fetch: bool = True) -> dict[tuple[int, str], list[dict]]:
    """한 달치 공고 목록·기초금액 원시 행. 캐시(jsonl)가 있으면 그것을 쓴다(개인정보 필드는 저장하지 않는다)."""
    bgn, end = cg.month_bounds(ym)
    out = {}
    for (wt, kind), op in OPS.items():
        path = cache_dir / f"{op}_{ym}.jsonl"
        if path.exists():
            out[(wt, kind)] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            continue
        if not fetch:
            raise FileNotFoundError(f"캐시 없음: {path} (--no-fetch 없이 실행)")
        client = client or BN.make_client()
        rows, page = [], 1
        while True:
            body = client.get(op, {"inqryDiv": 1, "inqryBgnDt": bgn, "inqryEndDt": end,
                                   "pageNo": page, "numOfRows": cg.PAGE_SIZE})
            rows.extend({k: it.get(k) for k in KEEP if k in it} for it in body["items"])
            total = int(body.get("totalCount") or 0)
            print(f"  {op} {ym} p{page}: 누적 {len(rows):,}/{total:,}", file=sys.stderr)
            if page * cg.PAGE_SIZE >= total or not body["items"]:
                break
            page += 1
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        out[(wt, kind)] = rows
    return out


def build_table(raw: dict[tuple[int, str], list[dict]]) -> pd.DataFrame:
    """공고(최신 차수) 1건 1행 + 기초금액(같은 차수, 분류별 기초금액이 하나일 때) 조인 + 규칙표 판정."""
    frames = []
    for wt in (1, 2):
        n = pd.DataFrame(raw.get((wt, "notice"), []))
        if n.empty:
            continue
        n = n.sort_values("bidNtceOrd").drop_duplicates("bidNtceNo", keep="last")
        b = pd.DataFrame(raw.get((wt, "bsis"), []))
        if not b.empty:
            b = b.drop_duplicates(["bidNtceNo", "bidNtceOrd", "bssamt"])
            b = b[~b.duplicated(["bidNtceNo", "bidNtceOrd"], keep=False)]    # 분류별 기초금액이 여러 개면 제외
            n = n.merge(b[["bidNtceNo", "bidNtceOrd", "bssamt", "rsrvtnPrceRngBgnRate", "rsrvtnPrceRngEndRate"]],
                        on=["bidNtceNo", "bidNtceOrd"], how="left")
        frames.append(n.assign(work_type=wt))
    df = pd.concat(frames, ignore_index=True)
    df["agency"] = df["dminsttNm"].fillna("").where(df["dminsttNm"].fillna("") != "", df["ntceInsttNm"])
    df["est_price"] = pd.to_numeric(df["presmptPrce"], errors="coerce")
    df["notice_lower"] = pd.to_numeric(df["sucsfbidLwltRate"], errors="coerce")
    df["notice_date"] = df["bidNtceDt"].astype(str).str[:10]
    df["method"] = df["sucsfbidMthdNm"].map(method_category)
    df["servc_kind"] = df["srvceDivNm"].map(BN.SERVC_KIND) if "srvceDivNm" in df else None
    lo = pd.to_numeric(df.get("rsrvtnPrceRngBgnRate"), errors="coerce")
    hi = pd.to_numeric(df.get("rsrvtnPrceRngEndRate"), errors="coerce")
    df["notice_range"] = hi.where((lo.abs() == hi.abs()) & hi.abs().isin([2, 3])).abs()

    rule_lower, rule_id, rule_range, regime = [], [], [], []
    for r in df.itertuples():
        sk = r.servc_kind if isinstance(r.servc_kind, str) else None
        lw = R.resolve_lower_limit(r.agency, work_type=r.work_type, est_price=r.est_price if pd.notna(r.est_price) else 0,
                                   as_of=r.notice_date, servc_kind=sk) if r.agency else None
        rg = R.resolve_range(r.agency, work_type=r.work_type, as_of=r.notice_date) if r.agency else None
        rule_lower.append(lw["lower_limit_pct"] if lw else None)
        rule_id.append(lw["rule_id"] if lw else None)
        rule_range.append(rg["range_pct"] if rg else None)
        regime.append(rg["rule_id"] if rg else "미분류")
    df["rule_lower"], df["rule_lower_id"], df["rule_range"], df["regime"] = rule_lower, rule_id, rule_range, regime
    return df


def summarize(df: pd.DataFrame) -> dict:
    """하한율·예가 범위 비교 요약. 하한율은 공고값이 있는 건만, 예가 범위는 기초금액이 조인된 건만."""
    d = df[df["notice_lower"].between(50, 100)]
    judged = d[d["rule_lower"].notna()]
    match = (judged["rule_lower"] - judged["notice_lower"]).abs() < 1e-6
    by_method = (judged.assign(match=match).groupby("method")["match"].agg(["size", "mean"])
                 .rename(columns={"size": "n", "mean": "match_rate"}).sort_values("n", ascending=False))
    by_rule = (judged.assign(match=match).groupby("rule_lower_id")["match"].agg(["size", "mean"])
               .rename(columns={"size": "n", "mean": "match_rate"}).sort_values("n", ascending=False))
    q = judged[judged["method"] == "적격심사"]
    qm = (q["rule_lower"] - q["notice_lower"]).abs() < 1e-6
    mism = q[~qm]
    mism_pairs = Counter(zip(mism["rule_lower_id"], mism["rule_lower"], mism["notice_lower"])).most_common(15)
    unjudged = d[d["rule_lower"].isna() & (d["method"] == "적격심사")]
    rg = df[df["notice_range"].notna() & df["rule_range"].notna()]
    rmatch = rg["notice_range"] == rg["rule_range"]
    return {
        "n_notice": int(len(df)), "n_with_lower": int(len(d)), "n_judged": int(len(judged)),
        "match_rate_all": float(match.mean()) if len(judged) else None,
        "n_qualified": int(len(q)), "match_rate_qualified": float(qm.mean()) if len(q) else None,
        "by_method": by_method, "by_rule": by_rule, "mismatch_pairs": mism_pairs,
        "mismatch_examples": mism.head(15)[["bidNtceNo", "agency", "work_type", "est_price", "notice_date", "servc_kind",
                                             "rule_lower_id", "rule_lower", "notice_lower", "sucsfbidMthdNm"]],
        "unjudged_qualified": int(len(unjudged)),
        "unjudged_top_regime": Counter(unjudged["regime"]).most_common(8),
        "n_range": int(len(rg)), "range_match_rate": float(rmatch.mean()) if len(rg) else None,
        "range_by_regime": (rg.assign(match=rmatch).groupby("regime")["match"].agg(["size", "mean"])
                            .rename(columns={"size": "n", "mean": "match_rate"}).sort_values("n", ascending=False)),
    }


def _pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x*100:.2f}%"


def write_report(s: dict, ym: str, path: Path) -> str:
    L = [f"# 규칙표 vs 입찰공고 검증 ({ym} 공고)", "",
         "- 하한율: 공고 `sucsfbidLwltRate` vs `rules.resolve_lower_limit` (발주기관·공종·추정가격·공고일·용역구분)",
         "- 예가 범위: 기초금액 조회 `rsrvtnPrceRng*` vs `rules.resolve_range`", "",
         f"공고 {s['n_notice']:,}건 중 하한율 기재 {s['n_with_lower']:,}건, 규칙표 판정 {s['n_judged']:,}건 → 일치율 {_pct(s['match_rate_all'])}",
         f"**적격심사 {s['n_qualified']:,}건 일치율 {_pct(s['match_rate_qualified'])}** (규칙표는 적격심사 통상 하한율만 담는다)", "",
         "## 낙찰방법별", "", "| 낙찰방법 | 판정 건수 | 일치율 |", "|---|---:|---:|"]
    L += [f"| {k} | {int(v.n):,} | {_pct(v.match_rate)} |" for k, v in s["by_method"].iterrows()]
    L += ["", "## 규칙 행별 (전 낙찰방법)", "", "| 규칙 ID | 판정 건수 | 일치율 |", "|---|---:|---:|"]
    L += [f"| `{k}` | {int(v.n):,} | {_pct(v.match_rate)} |" for k, v in s["by_rule"].iterrows()]
    L += ["", "## 적격심사 불일치 유형 (규칙 ID, 규칙값, 공고값, 건수)", "", "| 규칙 ID | 규칙표 | 공고 | 건수 |", "|---|---:|---:|---:|"]
    L += [f"| `{rid}` | {rv} | {nv} | {c:,} |" for (rid, rv, nv), c in s["mismatch_pairs"]]
    L += ["", "### 불일치 예시", "", s["mismatch_examples"].to_markdown(index=False) if len(s["mismatch_examples"]) else "(없음)", "",
          f"## 규칙표가 판정하지 못한 적격심사 공고: {s['unjudged_qualified']:,}건", "",
          "| 예가 범위 기관 분류 | 건수 |", "|---|---:|"]
    L += [f"| {k} | {c:,} |" for k, c in s["unjudged_top_regime"]]
    L += ["", f"## 예가 범위: {s['n_range']:,}건 일치율 {_pct(s['range_match_rate'])}", "",
          "| 기관 분류 | 건수 | 일치율 |", "|---|---:|---:|"]
    L += [f"| {k} | {int(v.n):,} | {_pct(v.match_rate)} |" for k, v in s["range_by_regime"].iterrows()]
    text = "\n".join(L) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="규칙표(하한율·예가 범위) vs 입찰공고 대량 검증")
    ap.add_argument("--month", required=True, help="공고 등록월 YYYY-MM")
    ap.add_argument("--no-fetch", action="store_true", help="API 호출 없이 out/notice_raw 캐시만 사용")
    args = ap.parse_args(argv)
    try:
        raw = fetch_month(args.month, fetch=not args.no_fetch)
    except (FileNotFoundError, ValueError, cg.ApiError, cg.DailyLimitReached) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    s = summarize(build_table(raw))
    path = OUT_DIR / f"notice_rules_check_{args.month}.md"
    print(write_report(s, args.month, path))
    print(f"리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
