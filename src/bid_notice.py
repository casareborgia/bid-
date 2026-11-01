"""나라장터 입찰공고정보서비스로 공고번호 → 예측 입력값 조회.

공고 1건에서 예측에 필요한 값을 모두 얻는다 (2026-11-01 실조회로 필드 확인, tests/fixtures/g2b/ 녹화본).
  공고 목록  getBidPblancListInfo{Cnstwk,Servc}          : 추정가격 presmptPrce, 낙찰하한율 sucsfbidLwltRate, 수요기관 dminsttNm,
                                                          공고일 bidNtceDt, 예비가격 수 totPrdprcNum/drwtPrdprcNum, 용역구분 srvceDivNm
  기초금액   getBidPblancListInfo{Cnstwk,Servc}BsisAmount : 기초금액 bssamt, 예가 범위 rsrvtnPrceRng{Bgn,End}Rate,
                                                          A값 항목(입찰가격산식A정보조회와 같은 값), A값 적용 여부 bidPrceCalclAYn
조회는 inqryDiv=2 + bidNtceNo. 공사 목록에 없으면 용역 목록을 본다(최대 3호출).
기초금액은 개찰 전 공개되므로 미공개 단계면 base_amt/range_pct 가 None 이고 경고를 남긴다.

사용 예
  uv run python src/bid_notice.py R26BK01752065
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_g2b_api import ApiError, G2BClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://apis.data.go.kr/1230000/ad/BidPublicInfoService"
NOTICE_OP = {1: "getBidPblancListInfoCnstwk", 2: "getBidPblancListInfoServc"}
BSIS_OP = {1: "getBidPblancListInfoCnstwkBsisAmount", 2: "getBidPblancListInfoServcBsisAmount"}
# 입찰가격산식A정보조회(getBidPblancListBidPrceCalclAInfo)의 7개 항목. 품질관리비는 qltyMngcstAObjYn=Y 일 때만 A값에 넣는다
A_ITEMS = {
    "npnInsrprm": "국민연금보험료",
    "mrfnHealthInsrprm": "국민건강보험료",
    "odsnLngtrmrcprInsrprm": "노인장기요양보험료",
    "rtrfundNon": "퇴직공제부금",
    "sftyMngcst": "산업안전보건관리비",
    "sftyChckMngcst": "안전관리비",
    "qltyMngcst": "품질관리비",
}
SERVC_KIND = {"일반용역": "general", "기술용역": "tech"}


def _num(v) -> float | None:
    try:
        s = str(v).strip().replace(",", "")
        return float(s) if s else None
    except ValueError:
        return None


def _latest(items: list[dict]) -> dict | None:
    """정정공고가 있으면 차수(bidNtceOrd)가 가장 큰 것."""
    return max(items, key=lambda x: str(x.get("bidNtceOrd", ""))) if items else None


def a_value_from(item: dict) -> tuple[float, dict[str, float]]:
    """기초금액(또는 A정보) 응답 1건에서 A값 합계와 항목별 금액."""
    if str(item.get("bidPrceCalclAYn", "Y")).upper() == "N":
        return 0.0, {}
    parts = {}
    for f, name in A_ITEMS.items():
        if f == "qltyMngcst" and str(item.get("qltyMngcstAObjYn", "Y")).upper() != "Y":
            continue
        v = _num(item.get(f))
        if v:
            parts[name] = v
    return float(sum(parts.values())), parts


def parse_notice(notice: dict, bsis: dict | None, work_type: int) -> dict:
    """공고 목록 1건 + 기초금액 1건(없으면 None) → 예측 입력값 dict."""
    warnings = []
    method = notice.get("prearngPrceDcsnMthdNm", "")
    n_prices, n_drawn = _num(notice.get("totPrdprcNum")), _num(notice.get("drwtPrdprcNum"))
    if method and method != "복수예가":
        warnings.append(f"예정가격 결정방법이 '{method}' 입니다. 모델은 복수예가(15개 중 4개 추첨)만 학습했습니다")
    elif (n_prices, n_drawn) not in ((15, 4), (None, None)):
        warnings.append(f"예비가격 {n_prices:.0f}개 중 {n_drawn:.0f}개 추첨 — 모델은 15개 중 4개만 학습했습니다")

    base_amt = range_pct = None
    a_value, a_items = 0.0, {}
    if bsis:
        base_amt = _num(bsis.get("bssamt"))
        lo, hi = _num(bsis.get("rsrvtnPrceRngBgnRate")), _num(bsis.get("rsrvtnPrceRngEndRate"))
        if lo is not None and hi is not None:
            if abs(lo) == abs(hi) and abs(hi) in (2, 3):
                range_pct = int(abs(hi))
            else:
                warnings.append(f"예가 범위 {lo:+g}%~{hi:+g}% 는 모델이 학습한 ±2/±3% 가 아닙니다")
        if work_type == 1:
            a_value, a_items = a_value_from(bsis)
    else:
        warnings.append("기초금액 미공개(또는 조회 실패): 기초금액은 추정가격×1.1, 예가 범위는 발주기관 규칙표로 대신합니다")

    lower = _num(notice.get("sucsfbidLwltRate"))
    return {
        "bid_no": notice.get("bidNtceNo"),
        "bid_ord": notice.get("bidNtceOrd"),
        "name": notice.get("bidNtceNm"),
        "work_type": work_type,
        "notice_date": str(notice.get("bidNtceDt", ""))[:10] or None,
        "agency": notice.get("dminsttNm") or notice.get("ntceInsttNm"),
        "est_price": _num(notice.get("presmptPrce")),
        "base_amt": base_amt,
        "range_pct": range_pct,
        "lower_limit_pct": lower if lower and 50 < lower < 100 else None,
        "a_value": a_value,
        "a_items": a_items,
        "servc_kind": SERVC_KIND.get(notice.get("srvceDivNm", "")) if work_type == 2 else None,
        "award_method_name": notice.get("sucsfbidMthdNm"),
        "contract_method": notice.get("cntrctCnclsMthdNm"),
        "open_dt": notice.get("opengDt"),
        "warnings": warnings,
    }


def make_client() -> G2BClient:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("G2B_SERVICE_KEY")
    if not key:
        raise ValueError(".env 에 G2B_SERVICE_KEY 가 없습니다")
    # 단건 조회라 상태 파일(out/collect_state.json)을 쓰지 않는다
    return G2BClient(key, state={"date": "", "calls_today": 0, "tasks": {}}, base_url=BASE_URL)


def fetch_notice(bid_no: str, client: G2BClient | None = None) -> dict:
    """공고번호로 예측 입력값을 조회한다. 없으면 LookupError, API 오류는 ApiError."""
    bid_no = str(bid_no).strip()
    client = client or make_client()
    q = {"inqryDiv": "2", "bidNtceNo": bid_no, "numOfRows": "50", "pageNo": "1"}
    for work_type in (1, 2):
        notice = _latest(client.get(NOTICE_OP[work_type], q)["items"])
        if notice:
            break
    else:
        raise LookupError(f"공고번호 {bid_no} 를 나라장터 공사·용역 입찰공고에서 찾지 못했습니다")
    rows = [x for x in client.get(BSIS_OP[work_type], q)["items"] if x.get("bidNtceOrd") == notice.get("bidNtceOrd")]
    if len({x.get("bssamt") for x in rows}) > 1:
        raise ValueError(f"공고 {bid_no} 는 분류(bidClsfcNo)별 기초금액이 여러 개입니다. --est_price 등으로 직접 입력하세요")
    return parse_notice(notice, rows[0] if rows else None, work_type)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("사용법: uv run python src/bid_notice.py <공고번호>", file=sys.stderr)
        return 2
    try:
        print(json.dumps(fetch_notice(argv[0]), ensure_ascii=False, indent=2))
    except (LookupError, ValueError, ApiError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
