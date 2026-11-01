"""실제 예정가격 대비 검증 (Phase 7).

세 가지 모드
  단건(공고번호): 나라장터 수집 parquet 에서 실제 기초금액·예정가격·예가 범위를 찾아 시장 모델 예측과 비교.
      parquet 에 없으면(수집 범위 밖) 낙찰정보 API 에서 실제값을, 입찰공고정보 API 에서 추정가격·하한율·A값을 조회한다.
      uv run python src/evaluate.py --bid_no 20240617004
  단건(수동):     추정가격과 실제 예정가격을 직접 넣어 비교 (사내 모델 기본, --market --range N 이면 시장 모델)
      uv run python src/evaluate.py --est_price 1000000000 --actual_yega 1098000000 --work_type 1
  배치(홀드아웃): 컷오프 이전 데이터로 모델을 다시 적합해 홀드아웃 월의 실제값과 비교 → out/backtest_<source>_<holdout>.md
      uv run python src/evaluate.py --holdout 2026-09 --source market --sample 10
      uv run python src/evaluate.py --holdout 2026-01,2026-10 --source internal

판정 규칙 (predict_cli 와 동일)
  Qα 기준 투찰가 = 예측 예정가격(Qα) × 낙찰하한율 × (1+안전마진). 실제 낙찰하한 = 실제 예정가격 × 낙찰하한율.
  투찰가 ≥ 실제 낙찰하한 이면 유효. 즉 실제 사정률 ≤ Qα×(1+마진) 일 때 유효하므로 유효율 ≈ α.
주의: 단건 모드에서 기본 모델은 그 입찰을 학습에 포함했을 수 있다(in-sample). 정직한 성적은 배치(홀드아웃) 모드로 본다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import predict_cli as P  # noqa: E402
import train_model as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G2B_PATH = ROOT / "data" / "g2b_detail_all.parquet"
INTERNAL_PATH = ROOT / "data" / "internal_clean_post2019.parquet"
OUT_DIR = ROOT / "out"
QUANTILES = (0.25, 0.50, 0.75)
DEFAULT_LOWER_LIMIT = 87.745
DEFAULT_MARGIN = 0.1


# ---------------------------------------------------------------- 단건
def lookup_bid(bid_no: str, parquet: Path | None = None) -> pd.DataFrame:
    """공고번호로 수집 parquet 에서 실제값을 찾는다 (차수·분류가 여러 개면 여러 행). parquet 기본은 모듈의 G2B_PATH(호출 시점)."""
    g = pd.read_parquet(parquet or G2B_PATH)
    hit = g[g["bidNtceNo"].astype(str).str.strip() == str(bid_no).strip()].copy()
    return hit.sort_values("bid_key").reset_index(drop=True)


def _position(actual: float, q: dict[float, float]) -> str:
    if actual < q[0.25]:
        return "Q25 아래"
    if actual <= q[0.50]:
        return "Q25~Q50"
    if actual <= q[0.75]:
        return "Q50~Q75"
    return "Q75 위"


def compare_one(art: dict, *, est_price: float, actual_yega_amt: float, work_type: int = 2,
                range_cls: str | None = None, bidder_count: float | None = None,
                lower_limit: float = DEFAULT_LOWER_LIMIT, margin: float = DEFAULT_MARGIN, round_unit: int = 1000,
                as_of: str | None = None, dept: str | None = None, award_method: str = "적격심사",
                negotiated: bool = False, a_value: float = 0.0, notice: dict | None = None) -> dict:
    """한 건의 예측(Q25/Q50/Q75·기준별 투찰가)과 실제 예정가격을 비교한다. predict_cli.build_report 를 그대로 쓴다."""
    rng = int(range_cls.replace("±", "")) if range_cls else None
    args = argparse.Namespace(est_price=float(est_price), work_type=int(work_type), bidder_count=bidder_count,
                              lower_limit=lower_limit, award_method=award_method, dept=dept, as_of=as_of,
                              margin=margin, round_unit=round_unit, negotiated=negotiated, basis="Q75", range=rng,
                              a_value=a_value, notice=notice)
    rep = P.build_report(art, args)
    base_amt = rep["base_amt"]
    actual_rate = float(actual_yega_amt) / base_amt * 100.0
    actual_lower = P.lower_limit_amount(float(actual_yega_amt), lower_limit, a_value)
    q = {0.25: None, 0.5: None, 0.75: None}
    rows = []
    for x in rep["quantiles"]:
        a = {"Q25": 0.25, "Q50": 0.5, "Q75": 0.75}[x["quantile"]]
        q[a] = x["yega_rate"]
        valid = x["bid_amt"] >= actual_lower
        rows.append({
            "quantile": x["quantile"], "pred_rate": x["yega_rate"], "pred_yega_amt": x["yega_amt"],
            "bid_amt": x["bid_amt"], "nominal_pass_prob": x["pass_prob"],
            "valid": bool(valid), "bid_minus_actual_lower": int(round(x["bid_amt"] - actual_lower)),
        })
    return {
        "input": rep["input"], "model_mode": rep["model"]["mode"],
        "base_amt": base_amt,
        "actual": {"yega_amt": int(round(actual_yega_amt)), "yega_rate": round(actual_rate, 4),
                   "lower_limit_amt": int(round(actual_lower))},
        "predicted": {"Q25": q[0.25], "Q50": q[0.5], "Q75": q[0.75]},
        "abs_error_q50_pct": round(abs(actual_rate - q[0.5]), 4),
        "position": _position(actual_rate, q),
        "in_q25_q75": bool(q[0.25] <= actual_rate <= q[0.75]),
        "bids": rows,
    }


def fetch_actual(bid_no: str, client=None) -> pd.DataFrame:
    """수집 parquet 에 없는 공고의 실제값을 낙찰정보서비스에서 조회해 parquet 과 같은 행 형식으로 돌려준다.
    예비가격상세는 공고번호 조회가 inqryDiv=2(2026-11-01 실측, 4 는 '입력범위값 초과'), 개찰결과 목록은 inqryDiv=4. 공사→용역 순."""
    import collect_g2b_api as cg
    if client is None:
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        key = os.environ.get("G2B_SERVICE_KEY")
        if not key:
            raise ValueError(".env 에 G2B_SERVICE_KEY 가 없습니다")
        client = cg.G2BClient(key, state={"date": "", "calls_today": 0, "tasks": {}})
    no = str(bid_no).strip()
    for work, W in (("cnstwk", "Cnstwk"), ("servc", "Servc")):
        items = client.get(f"getOpengResultListInfo{W}PreparPcDetail",
                           {"inqryDiv": 2, "bidNtceNo": no, "pageNo": 1, "numOfRows": cg.PAGE_SIZE})["items"]
        if not items:
            continue
        listing = client.get(f"getOpengResultListInfo{W}",
                             {"inqryDiv": 4, "bidNtceNo": no, "pageNo": 1, "numOfRows": cg.PAGE_SIZE})["items"]
        d = pd.DataFrame(items).assign(_work=work, _ym="", _page=1)
        return cg.build_detail(d, pd.DataFrame(listing) if listing else None)
    return pd.DataFrame()


def evaluate_bid_no(bid_no: str, art: dict | None = None, parquet: Path | None = None, live: bool = True,
                    client=None, notice_client=None, lower_limit: float | None = None, **kw) -> list[dict]:
    """공고번호 한 건(차수별 여러 행 가능)을 시장 모델로 비교. 반환은 행별 결과 목록.
    parquet 에 없고 live=True 면 API 로 실제값과 공고값(추정가격·기초금액·하한율·A값)을 조회한다.
    lower_limit=None 이면 공고 하한율, 공고가 없으면 DEFAULT_LOWER_LIMIT."""
    hit, source, notice = lookup_bid(bid_no, parquet), "parquet", None
    if hit.empty and live:
        hit, source = fetch_actual(bid_no, client), "api"
        if not hit.empty:
            import bid_notice as BN
            try:
                notice = BN.fetch_notice(bid_no, client=notice_client)
            except (LookupError, ValueError, BN.ApiError):
                notice = None
    if hit.empty:
        raise LookupError(f"공고번호 {bid_no} 를 수집 데이터(2024-01~2026-09)"
                          + ("와 낙찰정보 API 모두에서 찾지 못했습니다 (개찰 전이거나 단일예가)" if live else "에서 찾지 못했습니다"))
    art = art or P.load_model(P.MARKET_MODEL_PATH)
    ll = lower_limit if lower_limit is not None else (notice or {}).get("lower_limit_pct") or DEFAULT_LOWER_LIMIT
    out = []
    for _, r in hit.iterrows():
        rc = r.get(F.RANGE_COL)
        if rc not in F.RANGE_CLASSES:
            out.append({"bid_key": r["bid_key"], "skipped": f"예가 범위 미상/비표준({rc})"})
            continue
        use_notice = notice if notice and notice.get("base_amt") == float(r["base_amt"]) else None
        res = compare_one(art, est_price=(use_notice or {}).get("est_price") or float(r["base_amt"]) / P.VAT,
                          actual_yega_amt=float(r["yega_amt"]),
                          work_type=int(r["work_type"]), range_cls=str(rc),
                          bidder_count=None if pd.isna(r.get("bidder_count")) else float(r["bidder_count"]),
                          as_of=str(pd.Timestamp(r["open_dt"]).date()) if pd.notna(r.get("open_dt")) else None,
                          lower_limit=ll, a_value=float((use_notice or {}).get("a_value") or 0) if int(r["work_type"]) == 1 else 0.0,
                          notice=use_notice, **kw)
        res["bid"] = {"bid_key": r["bid_key"], "bid_name": r.get("bid_name"), "agency": r.get("dminsttNm"),
                      "open_dt": str(r.get("open_dt")), "range_cls": str(rc), "work_type_txt": r.get("work_type_txt"),
                      "bidder_count": None if pd.isna(r.get("bidder_count")) else float(r["bidder_count"]),
                      "source": source, "notice_used": bool(use_notice)}
        out.append(res)
    return out


# ---------------------------------------------------------------- 배치(홀드아웃)
def _month_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m")


def expand_months(spec: str) -> list[str]:
    """'2026-09' | '2026-01,2026-03' | '2026-01~2026-10' → 월 목록."""
    out: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "~" in part:
            a, b = [x.strip() for x in part.split("~", 1)]
            y, m = map(int, a.split("-")); ye, me = map(int, b.split("-"))
            while (y, m) <= (ye, me):
                out.append(f"{y:04d}-{m:02d}"); m += 1
                if m > 12:
                    y, m = y + 1, 1
        else:
            out.append(part)
    return sorted(set(out))


def backtest(source: str, holdout: list[str], lower_limit: float = DEFAULT_LOWER_LIMIT, margin: float = DEFAULT_MARGIN,
             sample: int = 0, seed: int = 0, df_override: pd.DataFrame | None = None) -> dict:
    """컷오프(홀드아웃 첫 달 1일) 이전 데이터로 다시 적합 → 홀드아웃 월의 실제값과 비교.

    source="market": 피처 MARKET_FEATURES, 사전값·Baseline 은 예가 범위별. source="internal": 사내 경쟁입찰 모델과 동일.
    """
    holdout = sorted(holdout)
    cutoff = pd.Timestamp(holdout[0] + "-01")
    if df_override is not None:
        df = df_override.copy()
    elif source == "market":
        df, _ = T.load_market(G2B_PATH)
    else:
        raw = pd.read_parquet(INTERNAL_PATH)
        raw[F.DATE_COL] = pd.to_datetime(raw[F.DATE_COL])
        df, _ = T.split_negotiated(raw)
    df = T._prep_internal(df)
    df[F.DATE_COL] = pd.to_datetime(df[F.DATE_COL])
    is_test = _month_key(df[F.DATE_COL]).isin(holdout)
    is_train = df[F.DATE_COL] < cutoff
    if is_test.sum() == 0 or is_train.sum() < 100:
        raise ValueError(f"홀드아웃 {holdout} 행 {int(is_test.sum())}건 / 학습 {int(is_train.sum())}건: 검증 불가")
    features = F.MARKET_FEATURES if source == "market" else F.FEATURES
    cats = F.MARKET_CAT_FEATURES if source == "market" else F.CAT_FEATURES
    enc = F.fit_encoders(df[is_train])
    feat = F.build_features(df, enc)                       # 날짜순; 이동통계는 과거만 참조
    idx = feat["contract_no"]
    tr = idx.isin(df.loc[is_train, "contract_no"]).values
    te = idx.isin(df.loc[is_test, "contract_no"]).values
    Xtr, ytr = feat.loc[tr, features], feat.loc[tr, F.TARGET]
    Xte, yte = feat.loc[te, features], feat.loc[te, F.TARGET].values
    models, best = T.fit_quantile_models(Xtr, ytr, cat_features=cats)
    if source == "market":
        gp = T.make_group_priors(ytr, feat.loc[tr, F.RANGE_COL].astype(str))
        priors = T.priors_for_rows(gp, feat.loc[te, F.RANGE_COL])
        baseline = priors[0.5]
    else:
        pri = T.make_priors(ytr)
        priors = {a: np.full(len(yte), pri[a]) for a in QUANTILES}
        baseline = np.full(len(yte), float(ytr.mean()))
    p = T.predict_quantiles(models, Xte, priors)
    m = margin / 100.0

    def metrics(mask: np.ndarray) -> dict:
        y, n = yte[mask], int(mask.sum())
        if n == 0:
            return {"n": 0}
        d = {"n": n, "baseline_mae": float(np.mean(np.abs(y - baseline[mask]))),
             "ml_mae": float(np.mean(np.abs(y - p[0.5][mask]))),
             "coverage_q25_q75": float(np.mean((y >= p[0.25][mask]) & (y <= p[0.75][mask])))}
        for a in QUANTILES:
            k = f"q{int(a*100)}"
            d[f"valid_{k}"] = float(np.mean(y <= p[a][mask] * (1 + m)))          # 안전마진 포함 유효율
            d[f"valid_{k}_nomargin"] = float(np.mean(y <= p[a][mask]))
        return d

    res = {
        "source": source, "holdout": holdout, "cutoff": str(cutoff.date()),
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "lower_limit": lower_limit, "margin_pct": margin,
        "best_iter": best, "overall": metrics(np.ones(len(yte), bool)), "by_group": {},
    }
    group_col = F.RANGE_COL if source == "market" else "work_type"
    gvals = feat.loc[te, group_col].astype(str).values
    if source != "market":
        gvals = np.where(gvals == "1", "공사", "용역")
    for g in sorted(set(gvals)):
        res["by_group"][g] = metrics(gvals == g)
    wt = feat.loc[te, "work_type"].astype(int).values
    res["by_work_type"] = {int(w): metrics(wt == w) for w in sorted(set(wt))}
    if sample:
        rng = np.random.default_rng(seed)
        pick = rng.choice(np.where(te)[0], size=min(sample, int(te.sum())), replace=False)
        rows = []
        for i in sorted(pick):
            r = feat.iloc[i]
            j = int(np.where(np.where(te)[0] == i)[0][0])
            q = {a: float(p[a][j]) for a in QUANTILES}
            rows.append({"id": str(r["contract_no"])[:28], "date": str(pd.Timestamp(r[F.DATE_COL]).date()),
                         "group": str(r[group_col]) if source == "market" else ("공사" if int(r["work_type"]) == 1 else "용역"),
                         "work_type": int(r["work_type"]),
                         "est_price": float(r["est_price"]),
                         "q25": q[0.25], "q50": q[0.5], "q75": q[0.75], "actual": float(yte[j]),
                         "position": _position(float(yte[j]), q),
                         "valid_q75": bool(yte[j] <= q[0.75] * (1 + m))})
        res["sample"] = rows
    return res


def write_backtest_report(res: dict, path: Path) -> str:
    o = res["overall"]
    src = "시장 모델(나라장터)" if res["source"] == "market" else "사내 모델(경쟁입찰)"
    lines = [
        f"# 홀드아웃 백테스트 — {src}",
        "",
        f"- 생성: {dt.datetime.now():%Y-%m-%d %H:%M} | 홀드아웃 {', '.join(res['holdout'])} | 컷오프 {res['cutoff']} 이전 {res['n_train']:,}건으로 재적합 → 홀드아웃 {res['n_test']:,}건 비교",
        f"- 낙찰하한율 {res['lower_limit']}% | 안전마진 {res['margin_pct']}% | 조기종료 반복수 Q25/Q50/Q75 = {res['best_iter'][0.25]}/{res['best_iter'][0.5]}/{res['best_iter'][0.75]}",
        "- 유효율 = 해당 기준 투찰가가 실제 낙찰하한금액 이상인 비율 (= 실제 사정률 ≤ Qα×(1+마진)). 이상적: Q25 25%+, Q50 50%+, Q75 75%+ (마진만큼 조금 높음)",
        "",
        "## 전체",
        "",
        "| n | Baseline MAE | ML MAE | Q25~Q75 커버리지 | 유효율 Q25 | 유효율 Q50 | 유효율 Q75 | (마진 0) Q25/Q50/Q75 |",
        "|---|---|---|---|---|---|---|---|",
        f"| {o['n']:,} | {o['baseline_mae']:.4f} | {o['ml_mae']:.4f} | {o['coverage_q25_q75']*100:.1f}% | {o['valid_q25']*100:.1f}% | {o['valid_q50']*100:.1f}% | {o['valid_q75']*100:.1f}% | {o['valid_q25_nomargin']*100:.1f} / {o['valid_q50_nomargin']*100:.1f} / {o['valid_q75_nomargin']*100:.1f}% |",
        "",
        f"## {'예가 범위별' if res['source'] == 'market' else '공사/용역별'}",
        "",
        "| 그룹 | n | Baseline MAE | ML MAE | 커버리지 | 유효율 Q25 / Q50 / Q75 |",
        "|---|---|---|---|---|---|",
    ]
    for g, d in res["by_group"].items():
        if d["n"]:
            lines.append(f"| {g} | {d['n']:,} | {d['baseline_mae']:.4f} | {d['ml_mae']:.4f} | {d['coverage_q25_q75']*100:.1f}% | {d['valid_q25']*100:.1f} / {d['valid_q50']*100:.1f} / {d['valid_q75']*100:.1f}% |")
    if res["source"] == "market":
        lines += ["", "## 공사/용역별", "", "| work_type | n | ML MAE | 커버리지 | 유효율 Q75 |", "|---|---|---|---|---|"]
        for w, d in res["by_work_type"].items():
            if d["n"]:
                lines.append(f"| {'공사' if w == 1 else '용역'} | {d['n']:,} | {d['ml_mae']:.4f} | {d['coverage_q25_q75']*100:.1f}% | {d['valid_q75']*100:.1f}% |")
    if res.get("sample"):
        lines += ["", f"## 표본 {len(res['sample'])}건 (예측 vs 실제 사정률 %)", "",
                  "| 번호 | 날짜 | 그룹 | 추정가격 | Q25 | Q50 | Q75 | 실제 | 위치 | Q75 투찰 유효 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["sample"]:
            lines.append(f"| {r['id']} | {r['date']} | {r['group']} | {r['est_price']/1e8:.2f}억 | {r['q25']:.3f} | {r['q50']:.3f} | {r['q75']:.3f} | **{r['actual']:.3f}** | {r['position']} | {'유효' if r['valid_q75'] else '미달'} |")
    cov_ok = 0.45 <= o["coverage_q25_q75"] <= 0.55
    v75 = o["valid_q75_nomargin"]
    lines += ["", "## 판정", "",
              f"- 커버리지 {o['coverage_q25_q75']*100:.1f}% → 45~55%: {'통과' if cov_ok else '이탈'}",
              f"- Q75 기준 유효율(마진 0) {v75*100:.1f}% → 목표 75%±3: {'통과' if abs(v75-0.75) <= 0.03 else '이탈'}; 안전마진 {res['margin_pct']}% 적용 시 {o['valid_q75']*100:.1f}%",
              f"- ML MAE {o['ml_mae']:.4f} vs Baseline {o['baseline_mae']:.4f} ({(o['baseline_mae']-o['ml_mae'])/o['baseline_mae']*100:+.2f}%)", ""]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- 출력
def print_compare(res: dict) -> None:
    from rich.console import Console
    from rich.table import Table
    c = Console(width=max(Console().width, 110))
    b = res.get("bid")
    if b:
        c.print(f"\n[bold]{b['bid_key']}[/bold]  {b.get('bid_name') or ''}")
        c.print(f"[dim]{b.get('agency') or ''} · {b.get('work_type_txt') or ''} · 예가 범위 {b['range_cls']}% · 개찰 {b['open_dt'][:10]} · 참여 {b['bidder_count']}[/dim]")
    a = res["actual"]
    c.print(f"기초금액 {P.won(res['base_amt'])} | 실제 예정가격 [bold]{P.won(a['yega_amt'])}[/bold] (사정률 [bold]{a['yega_rate']:.3f}%[/bold]) | 실제 낙찰하한 {P.won(a['lower_limit_amt'])}")
    pr = res["predicted"]
    c.print(f"예측 사정률 Q25/Q50/Q75 = {pr['Q25']:.3f} / {pr['Q50']:.3f} / {pr['Q75']:.3f} → 실제는 [bold]{res['position']}[/bold], Q50 오차 {res['abs_error_q50_pct']:.3f}%p, 구간 적중 {'O' if res['in_q25_q75'] else 'X'}  [dim](모델 {res['model_mode']})[/dim]")
    t = Table(show_header=True, header_style="bold")
    for col, j in (("기준", "left"), ("투찰가", "right"), ("실제 하한 대비", "right"), ("명목 통과확률", "right"), ("결과", "left")):
        t.add_column(col, justify=j)
    for r in res["bids"]:
        t.add_row(r["quantile"], P.won(r["bid_amt"]), f"{r['bid_minus_actual_lower']:+,}원", f"약 {int(r['nominal_pass_prob']*100)}%",
                  "[green]유효[/green]" if r["valid"] else "[red]하한 미달[/red]")
    c.print(t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="예측 vs 실제 예정가격 검증")
    ap.add_argument("--bid_no", help="나라장터 공고번호 (수집 parquet, 없으면 낙찰정보·입찰공고 API 에서 실제값 조회, 시장 모델)")
    ap.add_argument("--est_price", type=float, help="수동 비교: 추정가격(원)")
    ap.add_argument("--actual_yega", type=float, help="수동 비교: 실제 예정가격(원)")
    ap.add_argument("--work_type", type=int, choices=[1, 2], default=2)
    ap.add_argument("--bidder_count", type=float, default=None)
    ap.add_argument("--market", action="store_true", help="수동 비교에서 시장 모델 사용 (--range 필요)")
    ap.add_argument("--range", type=int, choices=[2, 3], default=None)
    ap.add_argument("--lower_limit", type=float, default=None, help=f"낙찰하한율 %% (기본: --bid_no 는 공고값, 그 외 {DEFAULT_LOWER_LIMIT})")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--a_value", type=float, default=0.0, help="수동 비교: 공고문 A값(원, 공사만)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--holdout", help="배치 백테스트: 홀드아웃 월 YYYY-MM (쉼표 나열 또는 2026-01~2026-10 범위)")
    ap.add_argument("--source", choices=["market", "internal"], default="market")
    ap.add_argument("--sample", type=int, default=10, help="백테스트 리포트에 넣을 표본 건수")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.holdout:
        months = expand_months(args.holdout)
        res = backtest(args.source, months, lower_limit=args.lower_limit or DEFAULT_LOWER_LIMIT, margin=args.margin, sample=args.sample)
        tag = months[0] if len(months) == 1 else f"{months[0]}~{months[-1]}"
        path = OUT_DIR / f"backtest_{args.source}_{tag}.md"
        text = write_backtest_report(res, path)
        print(json.dumps({k: v for k, v in res.items() if k != "sample"}, ensure_ascii=False, indent=1, default=str) if args.json else text)
        print(f"\n리포트: {path}")
        return 0

    if args.bid_no:
        art = P.load_model(args.model or P.MARKET_MODEL_PATH)
        try:
            results = evaluate_bid_no(args.bid_no, art, lower_limit=args.lower_limit, margin=args.margin)
        except (LookupError, ValueError, RuntimeError) as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        else:
            for r in results:
                if "skipped" in r:
                    print(f"{r['bid_key']}: 건너뜀 ({r['skipped']})")
                else:
                    print_compare(r)
        return 0

    if args.est_price and args.actual_yega:
        if args.market and args.range is None:
            ap.error("--market 에는 --range {2,3} 가 필요합니다")
        art = P.load_model(args.model or (P.MARKET_MODEL_PATH if args.market else P.MODEL_PATH))
        res = compare_one(art, est_price=args.est_price, actual_yega_amt=args.actual_yega, work_type=args.work_type,
                          range_cls=f"±{args.range}" if args.range else None, bidder_count=args.bidder_count,
                          lower_limit=args.lower_limit or DEFAULT_LOWER_LIMIT, margin=args.margin, a_value=args.a_value)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str) if args.json else "") if args.json else print_compare(res)
        return 0

    ap.error("--bid_no, 또는 --est_price 와 --actual_yega, 또는 --holdout 중 하나가 필요합니다")
    return 2


if __name__ == "__main__":
    sys.exit(main())
