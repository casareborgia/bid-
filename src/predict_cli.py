"""추정가격 기반 예정가격·투찰가 예측 CLI (Phase 4).

사용 예
  uv run python src/predict_cli.py --est_price 1000000000 --work_type 2 --bidder_count 25 --lower_limit 87.745
  uv run python src/predict_cli.py --est_price 500000000 --award_method 협상 --dept OO본부 --json

계산 규칙
  기초금액 = 추정가격 × 1.1 (부가세 10%, 2019년 이후 무사정 원칙)
  예상 예정가격(Qα) = 기초금액 × 예측 사정률(Qα) / 100
  낙찰하한금액(Qα) = 예상 예정가격(Qα) × 낙찰하한율 / 100
  투찰 후보 = 낙찰하한금액(Qα) × (1 + 안전마진) 을 반올림 단위로 올림
    낙찰하한금액은 실제 예정가격에 비례한다. 실제 사정률이 Qα 보다 높게 나오면 하한선이 투찰가 위로 올라가
    하한 미달(무효)이 되므로, Qα 기준 투찰가는 **실제 사정률이 Qα 이하일 때만** 유효하다.
    - Q75 기준: 통과 확률 약 75% (안전, 기본 권장) — 투찰가는 가장 높다
    - Q50 기준: 약 50%
    - Q25 기준: 약 25% (공격적) — 통과하면 가장 낮은 가격이지만 4번 중 3번은 하한 미달
    안전마진은 투찰가를 하한 추정치 위로 올려 통과 확률을 조금 더 높인다(0.1% ≈ 사정률 0.1%p ≈ 수 %p).
    교차검증으로 측정한 기준별 실제 통과율은 out/cv_report.md 의 "기준별 투찰가 통과율"을 본다.
  --negotiated (수의계약): ML 모델 대신 학습에서 분리해 둔 수의계약 사정률의 경험 분포 Q25/Q50/Q75 를 쓴다.
    수의계약은 예가 추첨 절차가 아니어서 분포가 넓고 하방으로 치우친다.
  --market --range {2,3} (공공발주 전체, Phase 6): 사내 모델 대신 나라장터 시장 모델(models/market_model.joblib)을 쓴다.
    예가 범위(공고의 "기초금액 ±N% 범위 내 15개 예비가격")가 사정률 분포를 결정한다: ±2% std 0.50, ±3% std 0.74.
    참여업체수를 생략하면 시장 중앙값(약 30)을 쓴다. 부서·낙찰방법은 시장 모델에 없다.
  --bid_no <나라장터 공고번호>: 입찰공고정보서비스(bid_notice.py)로 추정가격·기초금액·예가 범위·낙찰하한율·A값·발주기관·공고일을
    채우고 시장 모델을 쓴다. 기초금액은 추정가격×1.1 대신 공고값을 쓴다. 직접 지정한 인자가 공고값보다 우선한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import rules as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "bidding_model.joblib"
MARKET_MODEL_PATH = ROOT / "models" / "market_model.joblib"
DEFAULT_BIDDERS_INTERNAL = 8      # 사내 중앙값
DEFAULT_LOWER_LIMIT = 87.745
VAT = 1.1
QUANTILES = (0.25, 0.50, 0.75)
# Qα 기준 투찰가의 하한선 통과 확률 ≈ P(실제 사정률 ≤ Qα) = α. (안전마진 0 기준, 마진이 있으면 조금 높아진다)
PASS_PROB = {0.25: 0.25, 0.50: 0.50, 0.75: 0.75}
DEFAULT_BASIS = "Q75"          # 권장 기준: 약 75% 확률로 유효한 안전한 투찰가
WORK_TYPE_NAME = {1: "공사", 2: "용역"}


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}  (먼저 uv run python src/train_model.py 실행)")
    return joblib.load(path)


def lower_limit_amount(yega: float, lower_limit_pct: float, a_value: float = 0.0) -> float:
    """낙찰하한금액. A값(공사의 사후정산 항목 합계: 국민연금·건강보험·노인장기요양·퇴직공제부금·산업안전보건관리비 등)이 있으면
    A 만큼은 하한율을 곱하지 않는다: (예정가격 - A) × 하한율 + A. A=0 이면 예정가격 × 하한율과 같다."""
    return (yega - a_value) * lower_limit_pct / 100.0 + a_value


def ceil_to(x: float, unit: int) -> int:
    return int(math.ceil(x / unit) * unit) if unit > 1 else int(math.ceil(x))


def predict_rates(art: dict, *, work_type: int, dept_name: str | None, award_method: str | None,
                  est_price: float, bidder_count: float, as_of: pd.Timestamp, range_cls: str | None = None) -> dict:
    """사정률 Q25/Q50/Q75 (%) 와 참고 정보. 축소 규칙은 train_model.predict_quantiles 와 동일.

    시장 모델(art["source"]=="market")은 features=MARKET_FEATURES, range_cls 필수, 사전값은 범위별(priors_by_range).
    """
    is_market = art.get("source") == "market"
    features = art.get("features") or F.FEATURES
    hist = art.get("history_tail")
    if hist is not None and len(hist):
        hist = hist[pd.to_datetime(hist[F.DATE_COL]) <= as_of][F.TARGET]
    else:
        hist = pd.Series(dtype=float)
    if is_market:
        if range_cls not in art.get("range_classes", F.RANGE_CLASSES):
            raise ValueError(f"시장 모델은 --range 가 필요합니다 (예가 범위 %, {['2', '3']} 중 하나)")
        priors = art["priors_by_range"][range_cls]
    else:
        priors = art["priors"]
    row = F.make_inference_row(work_type=work_type, dept_name=dept_name, award_method=award_method,
                               est_price=est_price, bidder_count=bidder_count, as_of=as_of,
                               history_rates=hist, enc=art["encoders"], range_cls=range_cls, features=features)
    shrink = art["shrink_ml"]
    raw = {a: float(art["models"][a].predict(row)[0]) for a in QUANTILES}
    rates = {a: shrink * raw[a] + (1 - shrink) * priors[a] for a in QUANTILES}
    rates[0.25] = min(rates[0.25], rates[0.50])
    rates[0.75] = max(rates[0.75], rates[0.50])
    return {
        "rates": rates, "raw": raw, "priors": priors,
        "features": {k: (str(v) if not isinstance(v, (int, float, np.floating, np.integer)) else float(v))
                     for k, v in row.iloc[0].items()},
        "history_n": int(len(hist)),
        "dept_mapped": str(row["dept_code"].iloc[0]) if "dept_code" in row else None,
        "award_mapped": str(row["award_method"].iloc[0]) if "award_method" in row else None,
        "range_cls": range_cls,
    }


def negotiated_rates(art: dict) -> dict:
    """수의계약 경험 분포의 Q25/Q50/Q75. 아티팩트에 분포가 없으면 ValueError."""
    dist = art.get("negotiated_dist")
    if not dist:
        raise ValueError("모델 아티팩트에 수의계약 경험 분포가 없습니다 (train_model.py 를 다시 실행하세요)")
    q = dist["quantiles"]
    return {"rates": {a: float(q[a]) for a in QUANTILES}, "raw": None, "priors": None, "features": None,
            "history_n": 0, "dept_mapped": None, "award_mapped": None, "negotiated": dist}


def apply_notice(args: argparse.Namespace, notice: dict) -> None:
    """공고 조회값(bid_notice.fetch_notice)을 비어 있는 인자에만 채운다. 직접 지정한 인자가 우선한다.
    기초금액·예가 범위·낙찰하한율·A값은 args.notice 로 넘겨 build_report 가 우선순위(명시 > 공고 > 규칙표)를 정한다."""
    args.notice = notice
    for attr, key in (("est_price", "est_price"), ("work_type", "work_type"), ("agency", "agency"),
                      ("as_of", "notice_date"), ("servc_kind", "servc_kind")):
        if getattr(args, attr, None) is None and notice.get(key) is not None:
            setattr(args, attr, notice[key])


def build_report(art: dict, args: argparse.Namespace) -> dict:
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(dt.date.today())
    notice = getattr(args, "notice", None) or {}
    notice_lbl = f"공고 {notice.get('bid_no')}"
    base_amt = int(round(notice["base_amt"])) if notice.get("base_amt") else int(round(args.est_price * VAT))
    negotiated = bool(getattr(args, "negotiated", False))
    is_market = art.get("source") == "market"
    agency = getattr(args, "agency", None)
    raw_range = getattr(args, "range", None)
    range_source = None
    range_basis = None
    if raw_range is not None:
        range_cls = f"±{int(raw_range)}"
        range_source = "explicit"
        if agency:
            r_info = R.resolve_range(agency, work_type=args.work_type, as_of=as_of)
            if r_info and r_info["range_pct"] == int(raw_range):
                range_basis = r_info["basis_text"]
    elif is_market and notice.get("range_pct"):
        range_cls = f"±{int(notice['range_pct'])}"
        range_source = "notice"
        range_basis = f"{notice_lbl} 기초금액 ±{int(notice['range_pct'])}% (나라장터 입찰공고정보)"
    elif is_market and agency:
        r_info = R.resolve_range(agency, work_type=args.work_type, as_of=as_of)
        if r_info is None:
            raise ValueError(
                f"발주기관 '{agency}' 의 예가 범위를 자동 판정할 수 없습니다(미등록 또는 혼재 기관). "
                "--range {2,3} 을 공고문에서 확인해 직접 지정하세요."
            )
        range_cls = r_info["range_cls"]
        range_source = "auto"
        range_basis = r_info["basis_text"]
    else:
        range_cls = None
        if not is_market:
            r_info = R.resolve_range(agency or "한국남동발전", work_type=args.work_type, as_of=as_of)
            if r_info:
                range_basis = r_info["basis_text"]

    raw_lower = getattr(args, "lower_limit", None)
    lower_limit_source = "default"
    lower_limit_basis = None
    if raw_lower is not None:
        lower_limit_pct = float(raw_lower)
        lower_limit_source = "explicit"
    elif notice.get("lower_limit_pct"):
        lower_limit_pct = float(notice["lower_limit_pct"])
        lower_limit_source = "notice"
        lower_limit_basis = f"{notice_lbl} 낙찰하한율 (나라장터 입찰공고정보)"
        if agency:
            l_info = R.resolve_lower_limit(agency, work_type=args.work_type, est_price=args.est_price, as_of=as_of,
                                           servc_kind=getattr(args, "servc_kind", None))
            if l_info is not None and abs(float(l_info["lower_limit_pct"]) - lower_limit_pct) > 1e-9:
                lower_limit_basis += f" — 규칙표 판정 {l_info['lower_limit_pct']}%와 다름, 공고값 우선"
    elif agency or not is_market:
        # 사내 모델은 발주기관이 한국남동발전이다 (공사·용역 모두 조달청 적격심사 기준 준용)
        l_info = R.resolve_lower_limit(agency or "한국남동발전", work_type=args.work_type, est_price=args.est_price, as_of=as_of,
                                       servc_kind=getattr(args, "servc_kind", None))
        if l_info is not None:
            lower_limit_pct = float(l_info["lower_limit_pct"])
            lower_limit_source = "auto"
            lower_limit_basis = l_info["basis_text"]
        else:
            lower_limit_pct = DEFAULT_LOWER_LIMIT
            lower_limit_basis = (f"규칙표 판정 불가(예: 조달청 일반용역은 별표별로 다름, 지자체 공사 50억 이상) — 기본값 {DEFAULT_LOWER_LIMIT}%. "
                                 "공고문의 하한율을 --lower_limit 로 넣거나 --bid_no 를 쓰세요")
    else:
        lower_limit_pct = DEFAULT_LOWER_LIMIT

    bidder_count = args.bidder_count
    if bidder_count is None:
        bidder_count = float(art.get("bidder_median") or DEFAULT_BIDDERS_INTERNAL) if is_market else float(DEFAULT_BIDDERS_INTERNAL)
    if negotiated:
        if is_market:
            raise ValueError("--negotiated 는 사내 모델에서만 쓸 수 있습니다 (시장 모델에는 수의계약 분포가 없음)")
        pr = negotiated_rates(art)
    else:
        pr = predict_rates(art, work_type=args.work_type, dept_name=args.dept, award_method=args.award_method,
                           est_price=args.est_price, bidder_count=bidder_count, as_of=as_of, range_cls=range_cls)
    raw_a = getattr(args, "a_value", None)
    if raw_a is not None:
        a_value, a_source = float(raw_a), "explicit"
    else:
        a_value = float(notice.get("a_value") or 0) if int(args.work_type) == 1 else 0.0
        a_source = "notice" if a_value else None
    if a_value < 0:
        raise ValueError("A값은 0 이상이어야 합니다")
    if a_value > 0 and int(args.work_type) != 1:
        raise ValueError("A값은 공사(work_type=1)에만 적용됩니다")
    margin = args.margin / 100.0
    rows = []
    for a in QUANTILES:
        rate = round(pr["rates"][a], 4)          # 표시값과 계산값을 일치시킨다
        yega = base_amt * rate / 100.0
        if a_value >= yega:
            raise ValueError(f"A값({a_value:,.0f}원)이 예정가격 추정치({yega:,.0f}원) 이상입니다. 공고문의 A값을 확인하세요")
        lower = lower_limit_amount(yega, lower_limit_pct, a_value)
        bid = ceil_to(lower * (1 + margin), args.round_unit)
        rows.append({
            "quantile": f"Q{int(a*100)}",
            "yega_rate": rate,
            "yega_amt": int(round(yega)),
            "lower_limit_amt": int(round(lower)),
            "bid_amt": bid,
            "bid_vs_base_pct": round(bid / base_amt * 100, 3),
            "bid_vs_yega_pct": round(bid / yega * 100, 3),
            "pass_prob": PASS_PROB[a],
        })
    basis = str(getattr(args, "basis", DEFAULT_BASIS) or DEFAULT_BASIS).upper()
    rec = next(x for x in rows if x["quantile"] == basis)
    return {
        "input": {
            "est_price": int(args.est_price), "work_type": args.work_type,
            "work_type_name": WORK_TYPE_NAME.get(args.work_type, "?"),
            "bidder_count": bidder_count, "lower_limit_pct": lower_limit_pct, "a_value": int(round(a_value)),
            "servc_kind": getattr(args, "servc_kind", None),
            "lower_limit_source": lower_limit_source, "lower_limit_basis": lower_limit_basis,
            "award_method": args.award_method, "dept": args.dept, "agency": agency,
            "as_of": str(as_of.date()),
            "margin_pct": args.margin, "round_unit": args.round_unit,
            "negotiated": negotiated, "market": is_market, "range_cls": range_cls,
            "range_source": range_source, "range_basis": range_basis,
            "a_source": a_source, "base_amt_source": "notice" if notice.get("base_amt") else "est_price×1.1",
            "bid_no": notice.get("bid_no"),
        },
        "notice": None if not notice else {
            k: notice.get(k) for k in ("bid_no", "bid_ord", "name", "agency", "notice_date", "open_dt", "est_price",
                                       "base_amt", "range_pct", "lower_limit_pct", "a_value", "a_items", "servc_kind",
                                       "award_method_name", "contract_method", "warnings")
        },
        "base_amt": base_amt,
        "quantiles": rows,
        "recommended": {"basis": basis, "bid_amt": rec["bid_amt"], "pass_prob": rec["pass_prob"],
                        "bid_vs_base_pct": rec["bid_vs_base_pct"]},
        "model": {
            "mode": "negotiated" if negotiated else ("market" if is_market else "ml"),
            "trained_at": art["trained_at"], "n_train": art["n_train"], "data_range": art["data_range"],
            "n_negotiated": int(art.get("n_negotiated", 0) or 0),
            "train_mean": round(art["train_mean"], 4),
            "priors": None if pr["priors"] is None else {str(k): round(v, 4) for k, v in pr["priors"].items()},
            "raw_rates": None if pr["raw"] is None else {str(k): round(v, 4) for k, v in pr["raw"].items()},
            "history_n": pr["history_n"], "dept_mapped": pr["dept_mapped"], "award_mapped": pr["award_mapped"],
            "features": pr["features"],
            "negotiated_dist": None if not negotiated else {
                "n": pr["negotiated"]["n"], "mean": round(pr["negotiated"]["mean"], 4),
                "std": round(pr["negotiated"]["std"], 4),
                "quantiles": {str(k): round(v, 4) for k, v in pr["negotiated"]["quantiles"].items()},
            },
        },
    }


def won(x: float | int) -> str:
    return f"{int(x):,}원"


def print_report(r: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    c = Console(width=max(Console().width, 118))
    i = r["input"]
    bidders = int(i["bidder_count"]) if float(i["bidder_count"]).is_integer() else i["bidder_count"]
    c.print(f"\n[bold]예정가격·투찰가 예측 리포트[/bold]  (기준일 {i['as_of']})")
    n = r.get("notice")
    if n:
        c.print(f"[bold]{n['bid_no']}[/bold] {n['name']} | {n['agency']} | 공고 {n['notice_date']} · 개찰 {n['open_dt']} | {n['award_method_name']}")
        for w in n["warnings"]:
            c.print(f"[yellow]주의: {w}[/yellow]")
    base_lbl = "기초금액(공고)" if i.get("base_amt_source") == "notice" else "기초금액(×1.1)"
    award = "" if i.get("market") else f" · {i['award_method']}"     # 시장 모델은 낙찰방법을 쓰지 않는다(공고 낙찰방법은 위 줄)
    c.print(f"추정가격 {won(i['est_price'])} → {base_lbl} [bold]{won(r['base_amt'])}[/bold]  | "
            f"{i['work_type_name']}{award} · 참여 {bidders}개사 · 낙찰하한율 {i['lower_limit_pct']}%")
    if i.get("a_value"):
        src = " (공고)" if i.get("a_source") == "notice" else ""
        c.print(f"A값{src} [bold]{won(i['a_value'])}[/bold] 반영: 낙찰하한금액 = (예정가격 - A값) × 하한율 + A값")
        if n and i.get("a_source") == "notice" and n.get("a_items"):
            c.print("[dim]A값 항목: " + ", ".join(f"{k} {won(v)}" for k, v in n["a_items"].items()) + "[/dim]")
    m = r["model"]
    if m["mode"] == "negotiated":
        nd = m["negotiated_dist"]
        c.print(f"[bold yellow]수의계약 모드[/bold yellow]: ML 대신 수의계약 {nd['n']}건의 경험 분포 사용 "
                f"(사정률 평균 {nd['mean']}%, 표준편차 {nd['std']}%p). 입력한 참여업체수·부서·낙찰방법은 쓰지 않습니다.")
    elif m["mode"] == "market":
        src_lbl = {"auto": "자동 판정", "notice": "공고"}.get(i.get("range_source"), "사용자 지정")
        c.print(f"[bold cyan]시장 모델(나라장터, 공공발주 전체)[/bold cyan]: {m['n_train']:,}건 ({m['data_range'][0]}~{m['data_range'][1]}) 학습 | "
                f"예가 범위 [bold]{i['range_cls']}%[/bold] ({src_lbl}) 기준 사전값 Q25/Q50/Q75 {m['priors']['0.25']}/{m['priors']['0.5']}/{m['priors']['0.75']} | "
                f"부서·낙찰방법은 쓰지 않음")
        if i.get("range_basis"):
            c.print(f"[dim]예가 범위 근거: {i['range_basis']}[/dim]")
        if i.get("lower_limit_basis"):
            c.print(f"[dim]낙찰하한율 근거: {i['lower_limit_basis']}[/dim]")
    else:
        c.print(f"[dim]모델: 경쟁입찰 {m['n_train']:,}건 ({m['data_range'][0]}~{m['data_range'][1]}) 학습, 학습평균 {m['train_mean']}% | "
                f"부서 매핑 {m['dept_mapped']} / 낙찰방법 매핑 {m['award_mapped']} / 최근 이력 {m['history_n']}건[/dim]")
        if i.get("range_basis"):
            c.print(f"[dim]예가 범위 근거: {i['range_basis']}[/dim]")
        if i.get("lower_limit_basis"):
            c.print(f"[dim]낙찰하한율 근거: {i['lower_limit_basis']}[/dim]")
        if float(i["bidder_count"]) <= 1:
            c.print("[yellow]참고: 참여업체수 1은 사내 데이터에서 거의 전부 수의계약이었습니다. "
                    "수의계약이면 --negotiated 를 쓰세요(경쟁입찰 모델은 수의계약을 학습하지 않았습니다).[/yellow]")

    t = Table(show_header=True, header_style="bold")
    t.add_column("분위수"); t.add_column("사정률", justify="right"); t.add_column("예상 예정가격", justify="right")
    t.add_column("낙찰하한금액", justify="right"); t.add_column("투찰 후보", justify="right")
    t.add_column("기초 대비", justify="right"); t.add_column("통과확률", justify="right")
    rec = r["recommended"]
    for q in r["quantiles"]:
        style = "bold green" if q["quantile"] == rec["basis"] else ""
        t.add_row(q["quantile"], f"{q['yega_rate']:.3f}%", won(q["yega_amt"]), won(q["lower_limit_amt"]),
                  won(q["bid_amt"]), f"{q['bid_vs_base_pct']:.3f}%", f"약 {int(q['pass_prob']*100)}%", style=style)
    c.print(t)
    c.print(f"[bold green]{rec['basis']} 기준 투찰금액: {won(rec['bid_amt'])}[/bold green]  "
            f"({rec['basis']} 기준 낙찰하한 + 안전마진 {i['margin_pct']}%, {i['round_unit']:,}원 단위 올림, "
            f"기초금액 대비 {rec['bid_vs_base_pct']:.3f}%, 통과확률 약 {int(rec['pass_prob']*100)}%)")
    c.print("[dim]해석: 낙찰하한금액은 실제 예정가격이 높을수록 올라갑니다(A값이 있어도 같음). 실제 사정률이 기준 분위수보다 높게 나오면 하한선이 투찰가 위로 "
            "올라가 하한 미달(무효)이 되므로, Qα 기준 투찰가는 실제 사정률이 Qα 이하일 때만 유효합니다. "
            "Q75 기준 약 75%(안전), Q50 약 50%, Q25 약 25%(공격적, 통과 시 가격은 가장 낮음). --basis 로 기준을 바꿀 수 있습니다.[/dim]")
    c.print("[dim]주의: 이 값은 하한 통과 기준이지 낙찰확률을 최대로 하는 가격이 아닙니다. 참여업체가 많으면 1순위는 실제 하한 바로 위에서 "
            "정해지므로(예: 2026-10 서울시 전기공사 1,601개사, 1순위가 하한 +1,356원) Q75 투찰가는 대개 1순위가 되지 못합니다.[/dim]\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="추정가격 기반 예정가격·투찰가 예측")
    ap.add_argument("--bid_no", default=None,
                    help="나라장터 입찰공고번호. 추정가격·기초금액·예가 범위·낙찰하한율·A값·발주기관·공고일을 공고에서 채우고 시장 모델을 쓴다")
    ap.add_argument("--est_price", type=float, default=None, help="추정가격(부가세 제외, 원). --bid_no 가 없으면 필수")
    ap.add_argument("--work_type", type=int, choices=[1, 2], default=None, help="1=공사, 2=용역 (기본 2, --bid_no 면 공고값)")
    ap.add_argument("--bidder_count", type=float, default=None,
                    help="예상 참여업체수 (기본: 사내 모델 8, 시장 모델은 시장 중앙값 약 39)")
    ap.add_argument("--lower_limit", type=float, default=None, help="낙찰하한율 %% (기본: 규칙표 자동 판정 — 사내 모델은 한국남동발전, 시장 모델은 --agency. 판정 불가 시 87.745)")
    ap.add_argument("--award_method", default="적격심사", help="낙찰자결정방법 (적격심사/제한적최저가/최저가/협상)")
    ap.add_argument("--dept", default=None, help="발주부서명 (사내 모델만). 미지정 시 '기타'")
    ap.add_argument("--agency", default=None, help="발주기관명 (예: 서울특별시 강남구, 한국도로공사). 예가 범위·낙찰하한율 자동 판정")
    ap.add_argument("--as_of", default=None, help="기준일 YYYY-MM-DD (기본 오늘)")
    ap.add_argument("--servc_kind", choices=["general", "tech"], default=None,
                    help="용역 종류: general=일반용역(청소·경비·시설관리·학술연구), tech=기술용역(설계·감리·엔지니어링). 낙찰하한율 판정용, 생략 시 두 종류 중 높은 하한율(보수적)")
    ap.add_argument("--a_value", type=float, default=None,
                    help="A값(원, 공사만). 공고문의 국민연금·건강보험·노인장기요양·퇴직공제부금·산업안전보건관리비 등 합계. 하한금액 = (예정가격-A)×하한율+A")
    ap.add_argument("--margin", type=float, default=0.1, help="낙찰하한금액 위 안전마진 %% (기본 0.1)")
    ap.add_argument("--round_unit", type=int, default=1000, help="투찰금액 올림 단위 원 (기본 1000)")
    ap.add_argument("--basis", choices=["Q25", "Q50", "Q75"], default=DEFAULT_BASIS,
                    help="권장 투찰가 기준 분위수 (기본 Q75 = 통과확률 약 75%%. Q50 약 50%%, Q25 약 25%%)")
    ap.add_argument("--model", type=Path, default=None, help="모델 파일 (기본: 사내 models/bidding_model.joblib, --market 이면 models/market_model.joblib)")
    ap.add_argument("--market", action="store_true", help="나라장터 시장 모델 사용 (공공발주 전체). --range 또는 --agency 필요")
    ap.add_argument("--range", type=int, choices=[2, 3], default=None, help="예가 범위 %% (공고의 '기초금액 ±N%%'). --market 에서 --agency 미지정 시 필수")
    ap.add_argument("--negotiated", action="store_true",
                    help="수의계약: ML 대신 학습에서 분리한 수의계약 사정률 경험 분포의 Q25/Q50/Q75 를 사용")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    args = ap.parse_args(argv)
    if args.bid_no:
        import bid_notice as BN
        try:
            notice = BN.fetch_notice(args.bid_no)
        except (LookupError, ValueError, BN.ApiError) as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        args.market = True                          # 나라장터 공고는 시장 모델
        apply_notice(args, notice)
    if args.work_type is None:
        args.work_type = 2
    if args.est_price is None:
        ap.error("--est_price 또는 --bid_no 가 필요합니다")
    if args.est_price <= 0:
        ap.error("--est_price 는 양수여야 합니다")
    if (args.a_value or 0) > 0 and args.work_type != 1:
        ap.error("--a_value 는 공사(--work_type 1)에만 씁니다")
    if args.lower_limit is not None and not (50 < args.lower_limit < 100):
        ap.error("--lower_limit 는 50~100 사이 %% 값이어야 합니다")

    if args.market and args.range is None and not args.agency:
        ap.error("--market 에는 --range {2,3} 또는 --agency <발주기관명>이 필요합니다. 공고의 '기초금액 ±N%% 범위 내 15개 예비가격'의 N 입니다")
    if not args.market and args.range is not None:
        ap.error("--range 는 --market 에서만 씁니다 (사내 모델은 ±2%% 규정으로 고정)")
    model_path = args.model or (MARKET_MODEL_PATH if args.market else MODEL_PATH)
    art = load_model(model_path)
    if args.market and art.get("source") != "market":
        print(f"오류: {model_path} 는 시장 모델이 아닙니다 (train_model.py --source market 으로 생성)", file=sys.stderr)
        return 1
    try:
        report = build_report(art, args)
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
