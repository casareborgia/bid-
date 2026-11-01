"""LLM 인터페이스용 공용 도구 함수 (Phase 7). MCP 서버(mcp_server.py)와 Ollama 채팅(ollama_chat.py)이 같은 함수를 쓴다.

규칙: 숫자는 전부 predict_cli / evaluate 가 계산한다. LLM 은 질문을 해석해 이 함수를 호출하고 결과를 그대로 전달하는
인터페이스일 뿐이며 예측·계산을 직접 하지 않는다 (CLAUDE.md "딥러닝·LLM 사용 금지"는 예측 모델에 대한 규칙이다).
모든 함수는 JSON 직렬화 가능한 dict 를 돌려주고, 입력 오류는 예외 대신 {"error": ...} 로 돌려준다(LLM 이 되물을 수 있게).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E  # noqa: E402
import predict_cli as P  # noqa: E402
import rules as R  # noqa: E402


def _load(market: bool, model_path: str | None):
    path = Path(model_path) if model_path else (P.MARKET_MODEL_PATH if market else P.MODEL_PATH)
    art = P.load_model(path)
    if market and art.get("source") != "market":
        raise ValueError(f"{path} 는 시장 모델이 아닙니다")
    return art


def predict_bid(est_price: float | None = None, work_type: int | None = None, market: bool = False, range_pct: int | None = None,
                agency: str | None = None, bidder_count: float | None = None, lower_limit: float | None = None,
                award_method: str = "적격심사", dept: str | None = None, basis: str = "Q75", margin: float = 0.1,
                negotiated: bool = False, as_of: str | None = None, model_path: str | None = None,
                a_value: float | None = None, servc_kind: str | None = None, bid_no: str | None = None) -> dict:
    """추정가격(또는 나라장터 공고번호)으로 예정가격(사정률) Q25/Q50/Q75 와 기준별 투찰가·권장 투찰가를 계산한다.

    Args:
        est_price: 추정가격(부가세 제외, 원). 기초금액 = 추정가격 × 1.1. bid_no 가 있으면 생략 가능
        work_type: 1=공사, 2=용역. 생략 가능(기본 2, bid_no 면 공고값)
        market: True 면 나라장터 시장 모델(공공발주 전체), False 면 사내(남동발전) 모델
        range_pct: 예가 범위 %(2 또는 3). 공고문 "기초금액 ±N% 범위 내 15개 예비가격"의 N. agency 가 있으면 자동 판정하므로 생략 가능
        agency: 발주기관명 (예: 서울특별시 강남구, 경기도교육청, 한국도로공사). 지정하면 예가 범위와 적격심사 낙찰하한율을 규칙표에서 자동 판정한다
        bidder_count: 예상 참여업체수. 생략 시 사내 8, 시장 중앙값(39)
        lower_limit: 낙찰하한율 %. 생략 가능(기본 87.745 또는 agency 기반 자동 판정)
        award_method: 낙찰자결정방법 (적격심사/제한적최저가/최저가/협상). 사내 모델만 사용. 생략 가능(기본 적격심사), 사용자가 말하지 않으면 묻지 않는다
        dept: 발주부서명. 사내 모델만 사용. 생략 가능
        basis: 권장 투찰가 기준 분위수 Q25/Q50/Q75. 생략 가능(기본 Q75 ≈ 통과확률 75%, 안전). Q50 50%, Q25 25%(공격적)
        margin: 낙찰하한금액 위 안전마진 %. 생략 가능(기본 0.1)
        negotiated: True 면 수의계약 경험 분포 사용 (사내 모델만)
        as_of: 기준일 YYYY-MM-DD (기본 오늘)
        model_path: 모델 파일 경로 (기본값 사용 권장)
        a_value: A값(원). 공사 공고문의 사후정산 항목(국민연금·건강보험·노인장기요양·퇴직공제부금·산업안전보건관리비 등) 합계. 공사에만 적용, 생략 가능(기본 0). 하한금액 = (예정가격-A)×하한율+A
        servc_kind: 용역 종류 general(일반용역: 청소·경비·시설관리·학술연구) / tech(기술용역: 설계·감리·엔지니어링). 낙찰하한율 판정용, 생략 가능(생략 시 두 종류 중 높은 하한율)
        bid_no: 나라장터 입찰공고번호 (예: R26BK01752065). 주면 추정가격·기초금액·예가 범위·낙찰하한율·A값·발주기관을 공고에서 채우고 시장 모델을 쓴다. 이것만 있으면 된다
    """
    try:
        notice = None
        if bid_no:
            import bid_notice as BN
            try:
                notice = BN.fetch_notice(str(bid_no))
            except (LookupError, BN.ApiError) as e:
                return {"error": str(e)}
            market = True
        elif est_price in (None, "", 0):
            return {"error": "est_price(추정가격) 또는 bid_no(나라장터 공고번호)가 필요합니다"}
        if not market and range_pct is not None:
            return {"error": "range_pct 는 market=True 일 때만 씁니다 (사내 모델은 ±2% 규정 고정)"}
        if market and range_pct is not None and range_pct not in (2, 3):
            return {"error": "시장 모델은 range_pct(예가 범위 %, 2 또는 3)가 필요합니다. 공고문의 '기초금액 ±N%'의 N 을 확인하세요."}
        if market and range_pct is None and not agency and not notice:
            return {"error": "시장 모델은 range_pct(예가 범위 %, 2 또는 3) 또는 agency(발주기관명)가 필요합니다. 공고문의 '기초금액 ±N%'의 N 을 확인하세요."}
        if servc_kind not in (None, "", "general", "tech"):
            return {"error": "servc_kind 는 general(일반용역) 또는 tech(기술용역)"}
        if basis not in ("Q25", "Q50", "Q75"):
            return {"error": "basis 는 Q25/Q50/Q75 중 하나"}
        art = _load(market, model_path)
        args = argparse.Namespace(est_price=None if est_price in (None, "", 0) else float(est_price),
                                  work_type=None if work_type in (None, "") else int(work_type),
                                  bidder_count=None if bidder_count in (None, "", 0) else float(bidder_count),
                                  lower_limit=None if lower_limit in (None, "", 0) else float(lower_limit),
                                  award_method=award_method or "적격심사", dept=dept or None,
                                  agency=agency or None,
                                  as_of=as_of or None, margin=float(margin), round_unit=1000, negotiated=bool(negotiated),
                                  basis=basis, range=int(range_pct) if range_pct else None,
                                  a_value=None if a_value in (None, "", 0) else float(a_value), servc_kind=servc_kind or None)
        if notice:
            P.apply_notice(args, notice)
        if args.work_type is None:
            args.work_type = 2
        if args.a_value and args.work_type != 1:
            return {"error": "a_value(A값)는 공사(work_type=1)에만 씁니다"}
        rep = P.build_report(art, args)
        basis_line = ""
        if rep["input"].get("range_basis"):
            basis_line = f" 예가 범위 {rep['input']['range_cls'] or '±2'}% — 근거: {rep['input']['range_basis']}."
        rep["note"] = ("Qα 기준 투찰가는 실제 사정률이 Qα 이하일 때만 낙찰하한을 넘는다(통과확률 ≈ α). "
                       "Q75 가 안전, Q25 는 공격적. 사정률은 추첨 결과라 어떤 모델도 분위수 이상으로는 맞출 수 없다." + basis_line)
        return rep
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}


def predict_bid_simple(est_price: float, market: bool, range_pct: int | None = None, work_type: int = 2,
                       agency: str | None = None, a_value: float | None = None) -> dict:
    """추정가격만으로 예정가격 Q25/Q50/Q75 와 권장 투찰가(Q75 기준, 통과확률 약 75%)를 계산한다. 공고의 발주기관명(agency)을 주면 예가 범위와 낙찰하한율을 자동 판정하고, 모르면 range_pct 를 직접 준다. 다른 조건은 전부 기본값을 쓰므로 추가로 물을 것이 없다.

    Args:
        est_price: 추정가격(부가세 제외, 원)
        market: 타 기관(공공발주 전체) 입찰이면 true, 남동발전 사내 발주면 false
        range_pct: 예가 범위 %(2 또는 3). agency 가 없을 때 market=true 면 필수. 공고문 "기초금액 ±N% 범위 내 15개 예비가격"의 N
        work_type: 1=공사, 2=용역 (모르면 2)
        agency: 발주기관명(생략 가능). 지정하면 예가 범위를 자동 판정한다
        a_value: A값(원). 공사 공고문의 사후정산 항목(국민연금·건강보험·노인장기요양·퇴직공제부금·산업안전보건관리비 등) 합계. 공사에만 적용, 생략 가능(기본 0). 하한금액 = (예정가격-A)×하한율+A
    """
    return predict_bid(est_price=est_price, work_type=work_type, market=market, range_pct=range_pct, agency=agency,
                       a_value=a_value)


def predict_bid_by_notice(bid_no: str, basis: str = "Q75") -> dict:
    """나라장터 입찰공고번호 하나로 예정가격 Q25/Q50/Q75 와 권장 투찰가를 계산한다. 추정가격·기초금액·예가 범위·낙찰하한율·A값(공사)·발주기관을 공고에서 자동으로 가져오므로 추가로 물을 것이 없다.

    Args:
        bid_no: 나라장터 입찰공고번호 (예: R26BK01752065)
        basis: 권장 투찰가 기준 분위수 Q25/Q50/Q75. 생략 가능(기본 Q75 ≈ 통과확률 75%)
    """
    return predict_bid(bid_no=bid_no, basis=basis)


def evaluate_bid(bid_no: str, lower_limit: float | None = None, margin: float = 0.1, model_path: str | None = None) -> dict:
    """나라장터 공고번호로 실제 예정가격을 찾아 시장 모델 예측과 비교한다 (수집 데이터 2024-01~2026-09 53.9만 건, 그 밖은 API 실시간 조회).

    Args:
        bid_no: 나라장터 입찰공고번호 (예: 20240617004, R26BK01697785). 이것만 있으면 된다. 수집 범위 밖이면 API 로 실제값을 조회한다
        lower_limit: 낙찰하한율 %. 생략 가능(기본: 수집 범위 밖 공고는 공고값, 그 외 87.745). 사용자에게 묻지 않는다
        margin: 안전마진 %. 생략 가능(기본 0.1). 사용자에게 묻지 않는다
        model_path: 시장 모델 파일 경로. 생략(기본값)
    """
    try:
        art = _load(True, model_path)
        rows = E.evaluate_bid_no(str(bid_no), art, lower_limit=None if lower_limit in (None, "", 0) else float(lower_limit),
                                 margin=float(margin))
        return {"bid_no": str(bid_no), "n_rows": len(rows), "results": rows,
                "note": "기본 시장 모델은 이 입찰을 학습에 포함했을 수 있다(in-sample). 정직한 성적은 backtest 를 본다."}
    except (LookupError, ValueError, FileNotFoundError, RuntimeError) as e:
        return {"error": str(e)}


def compare_manual(est_price: float, actual_yega: float, work_type: int = 2, market: bool = False,
                   range_pct: int | None = None, bidder_count: float | None = None, lower_limit: float = 87.745,
                   margin: float = 0.1, model_path: str | None = None, a_value: float | None = None) -> dict:
    """추정가격과 실제 예정가격을 직접 넣어 예측이 얼마나 맞았는지 비교한다.

    Args:
        est_price: 추정가격(원)
        actual_yega: 실제 예정가격(원)
        work_type: 1=공사, 2=용역
        market: True 면 시장 모델 (range_pct 필수)
        range_pct: 예가 범위 % (2 또는 3)
        bidder_count: 참여업체수
        lower_limit: 낙찰하한율 %
        margin: 안전마진 %
        model_path: 모델 파일 경로
        a_value: A값(원). 공사 공고문의 사후정산 항목(국민연금·건강보험·노인장기요양·퇴직공제부금·산업안전보건관리비 등) 합계. 공사에만 적용, 생략 가능(기본 0). 하한금액 = (예정가격-A)×하한율+A
    """
    try:
        if market and range_pct not in (2, 3):
            return {"error": "시장 모델은 range_pct(2 또는 3)가 필요합니다"}
        art = _load(market, model_path)
        return E.compare_one(art, est_price=float(est_price), actual_yega_amt=float(actual_yega), work_type=int(work_type),
                             range_cls=f"±{int(range_pct)}" if range_pct else None,
                             bidder_count=None if bidder_count in (None, "", 0) else float(bidder_count),
                             lower_limit=float(lower_limit), margin=float(margin), a_value=float(a_value or 0))
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}


def backtest_summary(source: str = "market", holdout: str = "2026-09", sample: int = 5) -> dict:
    """컷오프 이전 데이터로 다시 적합해 홀드아웃 월의 실제값과 비교한 성적(커버리지·기준별 유효율·MAE)을 돌려준다. 수 초~수십 초 걸린다.

    Args:
        source: market(나라장터) 또는 internal(사내)
        holdout: 홀드아웃 월. '2026-09' 또는 '2026-01~2026-10' 또는 쉼표 나열
        sample: 함께 돌려줄 표본 건수
    """
    try:
        res = E.backtest(source, E.expand_months(holdout), sample=int(sample))
        res["best_iter"] = {str(k): v for k, v in res["best_iter"].items()}
        return res
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}


def model_info(internal_path: str | None = None, market_path: str | None = None) -> dict:
    """사내 모델과 시장 모델의 학습 정보(건수·기간·검증 성적·사전값)를 돌려준다."""
    out = {}
    for name, path in (("internal", Path(internal_path) if internal_path else P.MODEL_PATH),
                       ("market", Path(market_path) if market_path else P.MARKET_MODEL_PATH)):
        if not path.exists():
            out[name] = {"error": f"모델 파일 없음: {path}"}
            continue
        a = P.load_model(path)
        cv = a.get("cv_summary") or {}
        out[name] = {
            "trained_at": a.get("trained_at"), "n_train": a.get("n_train"), "data_range": a.get("data_range"),
            "features": a.get("features"), "train_mean": round(float(a["train_mean"]), 4),
            "cv": {k: round(float(cv[k]), 4) for k in ("baseline_mae_val", "ml_mae_val", "gap", "coverage_q25_q75", "pass_q25", "pass_q50", "pass_q75") if k in cv},
            "priors": {str(k): round(float(v), 4) for k, v in (a.get("priors") or {}).items()},
            "priors_by_range": {r: {str(k): round(float(v), 4) for k, v in d.items()} for r, d in (a.get("priors_by_range") or {}).items()} or None,
            "negotiated_dist": (lambda d: None if not d else {"n": d["n"], "mean": round(d["mean"], 3), "std": round(d["std"], 3),
                                                              "quantiles": {str(k): round(v, 3) for k, v in d["quantiles"].items()}})(a.get("negotiated_dist")),
        }
    out["rules"] = ("사정률 = 예정가격/기초금액. 15개 예비가격 중 4개 추첨의 평균이라 예측 불가능한 노이즈가 대부분이며, "
                    "모델의 가치는 보정된 분위수(통과확률)다. Qα 기준 투찰가의 통과확률 ≈ α.")
    return out


def rule_basis(agency: str, work_type: int = 2, est_price: float | None = None, as_of: str | None = None,
               servc_kind: str | None = None) -> dict:
    """발주기관명·공종·추정가격에 적용되는 예가 범위(±2%/±3%)와 적격심사 낙찰하한율 규칙 및 법적 근거(규정명·조문·시행일·법제처 링크)를 돌려준다.

    Args:
        agency: 발주기관명 (예: 한국남동발전, 서울특별시 강남구, 경기도교육청, 한국도로공사)
        work_type: 1=공사, 2=용역 (기본 2)
        est_price: 추정가격(원, 생략 가능). 지정하면 금액 구간별 낙찰하한율까지 함께 조회한다
        as_of: 기준일 YYYY-MM-DD (생략 가능, 기본 오늘)
        servc_kind: 용역 종류 general(일반용역: 청소·경비·시설관리·학술연구) / tech(기술용역: 설계·감리·엔지니어링). 낙찰하한율 판정용, 생략 가능(생략 시 두 종류 중 높은 하한율)
    """
    if not agency or not str(agency).strip():
        return {"error": "agency(발주기관명)를 입력하세요."}
    r_info = R.resolve_range(agency, work_type=int(work_type), as_of=as_of)
    r_unv = R.resolve_range(agency, work_type=int(work_type), as_of=as_of, include_unverified=True) if r_info is None else None
    l_info = R.resolve_lower_limit(agency, work_type=int(work_type), est_price=est_price, as_of=as_of, servc_kind=servc_kind)
    return {
        "agency": str(agency).strip(),
        "work_type": int(work_type),
        "est_price": None if est_price is None else float(est_price),
        "as_of": as_of,
        "range_rule": r_info,
        "unverified_candidate": r_unv,
        "lower_limit_rule": l_info,
        "default_lower_limit_pct": P.DEFAULT_LOWER_LIMIT,
        "note": (
            f"예가 범위 ±{r_info['range_pct']}% — 근거: {r_info['basis_text']}"
            if r_info
            else "자동 판정 대상(verified) 기관이 아닙니다. 공고문에서 기초금액 ±N% 범위를 직접 확인하세요."
        ),
    }


TOOLS = [predict_bid_simple, predict_bid_by_notice, predict_bid, evaluate_bid, compare_manual, backtest_summary, model_info, rule_basis]
DISPATCH = {f.__name__: f for f in TOOLS}
