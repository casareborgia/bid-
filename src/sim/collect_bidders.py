"""업체별 개찰 결과 수집 (시뮬레이터 2단계 데이터).

낙찰정보서비스 `getOpengResultListInfoOpengCompt`(공고번호 조회)는 입찰 1건의 모든 투찰을 준다:
투찰금액 bidprcAmt, 투찰률 bidprcrt, 순위 opengRank, 비고 rmrk(예: 낙찰하한선 미달), 고른 예비가격 번호 drwtNo1/drwtNo2.
업체명·대표자명은 저장하지 않고 사업자번호는 SHA-256 앞 12자리로만 남긴다(반복 입찰자 식별용).

대상: 수집 parquet 의 해당 월 개찰분 중 입찰공고 캐시(out/notice_raw, check_notice_rules.py)로 하한율이 확인되는 적격심사 건.
원시 행은 data/raw/g2b/compt_<월>.jsonl 에 append 하고, 진행 상태는 out/compt_state_<월>.json 에 둔다(이어받기).

사용 예
  uv run python src/sim/collect_bidders.py --month 2026-10 --work servc --max-calls 600
  uv run python src/sim/collect_bidders.py --jobs 2026-10:servc,2026-10:cnstwk,2026-09:servc,2026-09:cnstwk --max-calls 900
종료 코드: 0 전부 완료 / 2 한도 도달(다음 날 같은 명령으로 이어받기) / 1 오류.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import check_notice_rules as C  # noqa: E402
import collect_g2b_api as cg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OP = "getOpengResultListInfoOpengCompt"
KEEP = ["bidNtceNo", "bidNtceOrd", "bidClsfcNo", "rbidNo", "opengRank", "bidprcAmt", "bidprcrt", "rmrk",
        "drwtNo1", "drwtNo2", "bidprcDt"]


def anonymize(item: dict) -> dict:
    row = {k: item.get(k) for k in KEEP}
    biz = str(item.get("prcbdrBizno") or "").strip()
    row["bidder"] = hashlib.sha256(biz.encode()).hexdigest()[:12] if biz else None
    return row


def prev_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"


def notice_table(month: str) -> pd.DataFrame:
    """개찰월 공고와 그 전달 공고(전달에 공고하고 이번 달에 개찰한 건)를 합친 입찰공고 표. 캐시가 있는 달만."""
    frames = []
    for ym in (prev_month(month), month):
        try:
            frames.append(C.build_table(C.fetch_month(ym, fetch=False)))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"{month} 입찰공고 캐시가 없습니다 (check_notice_rules.py --month {month})")
    nt = pd.concat(frames, ignore_index=True)
    return nt.sort_values(["bidNtceNo", "bidNtceOrd"]).drop_duplicates("bidNtceNo", keep="last")


def targets(month: str, work: str, parquet: Path = cg.OUT_DIR / "g2b_detail_all.parquet") -> pd.DataFrame:
    """수집 대상 입찰(입찰키·공고번호·기초금액·실제 예정가격·예가 범위·하한율·A값). 공사는 A값이 확인된 건만."""
    g = pd.read_parquet(parquet)
    g = g[(g["ym"] == month) & (g["work_type"] == cg.WORK_CODE[work]) & g["range_cls"].isin(["±2", "±3"])]
    nt = notice_table(month)
    nt = nt[(nt["method"] == "적격심사") & nt["notice_lower"].between(50, 100) & nt["a_value"].notna()]
    t = g.merge(nt[["bidNtceNo", "notice_lower", "a_value"]], on="bidNtceNo", how="inner").drop_duplicates("bid_key")
    return t.sort_values("bid_key").reset_index(drop=True)


def collect(month: str, work: str, max_calls: int, client=None, raw_dir: Path = cg.RAW_DIR,
            state_path: Path | None = None) -> dict:
    state_path = state_path or (ROOT / "out" / f"compt_state_{month}_{work}.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": []}
    done = set(state["done"])
    t = targets(month, work)
    if client is None:
        load_dotenv(ROOT / ".env")
        client = cg.G2BClient(os.environ["G2B_SERVICE_KEY"], state={"date": "", "calls_today": 0, "tasks": {}},
                              daily_limit=max_calls)
    out_path = raw_dir / f"compt_{month}.jsonl"
    calls, limited = 0, False
    for no in t["bidNtceNo"].unique():
        if no in done:
            continue
        page, rows = 1, []
        try:
            while True:
                body = client.get(OP, {"inqryDiv": 4, "bidNtceNo": no, "pageNo": page, "numOfRows": cg.PAGE_SIZE})
                calls += 1
                rows += [anonymize(it) for it in body["items"]]
                if page * cg.PAGE_SIZE >= int(body.get("totalCount") or 0) or not body["items"]:
                    break
                page += 1
        except cg.DailyLimitReached:
            limited = True
            break
        cg.append_jsonl(out_path, rows, {"_work": work, "_ym": month})
        done.add(no)
        state["done"] = sorted(done)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state))
    return {"targets": int(t["bidNtceNo"].nunique()), "done": len(done), "calls": calls, "limited": limited}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="업체별 개찰 결과 수집 (시뮬레이터용)")
    ap.add_argument("--month", help="개찰월 YYYY-MM (--jobs 를 쓰면 생략)")
    ap.add_argument("--work", choices=["servc", "cnstwk"], default="servc")
    ap.add_argument("--jobs", help="월:업무 쉼표 나열, 순서대로 같은 호출 한도를 나눠 쓴다 (예: 2026-10:cnstwk,2026-09:servc)")
    ap.add_argument("--max-calls", type=int, default=600, help="이번 실행 호출 상한(일일 한도 950 안에서 다른 수집과 나눠 쓴다)")
    args = ap.parse_args(argv)
    jobs = [tuple(j.split(":")) for j in args.jobs.split(",")] if args.jobs else [(args.month, args.work)]
    if any(m is None for m, _ in jobs):
        ap.error("--month 또는 --jobs 가 필요합니다")
    load_dotenv(ROOT / ".env")
    client = cg.G2BClient(os.environ["G2B_SERVICE_KEY"], state={"date": "", "calls_today": 0, "tasks": {}},
                          daily_limit=args.max_calls)
    complete = True
    for month, work in jobs:
        try:
            res = collect(month, work, args.max_calls, client=client)
        except (cg.ApiError, FileNotFoundError, KeyError) as e:
            print(f"오류({month}:{work}): {e}", file=sys.stderr)
            return 1
        print(json.dumps({"job": f"{month}:{work}", **res}, ensure_ascii=False), flush=True)
        complete &= res["done"] >= res["targets"]
        if res["limited"]:
            return 2
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
