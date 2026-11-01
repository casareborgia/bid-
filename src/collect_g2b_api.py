"""조달청 나라장터 낙찰정보서비스 수집기 (Phase 2).

bssamt(기초금액)·plnprc(예정가격)는 *PreparPcDetail 오퍼레이션에만 있고,
참여업체수(prtcptCnum)·개찰일시(opengDt)·수요기관(dminsttNm)은 개찰결과 목록에 있다.
두 응답을 입찰키(bidNtceNo+bidNtceOrd+bidClsfcNo+rbidNo)로 조인한다.

모드
  market : 월 단위로 전체 시장을 수집 (기본). 월당 공사+용역 약 120~130 호출.
  agency : 수요기관명으로 낙찰목록현황을 조회한 뒤 공고번호별로 예비가격상세를 호출.

상태는 out/collect_state.json 에 저장되어 중단 후 재실행 시 이어받는다.
원시 응답은 data/raw/g2b/*.jsonl 에 페이지 단위로 append 되고, --build 로 parquet를 만든다.

사용 예
  uv run python src/collect_g2b_api.py --start 2026-09 --end 2026-09 --work servc
  uv run python src/collect_g2b_api.py --mode agency --agency 한국남동발전 --start 2024-01 --end 2026-09
  uv run python src/collect_g2b_api.py --build
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://apis.data.go.kr/1230000/as/ScsbidInfoService"
RAW_DIR = ROOT / "data" / "raw" / "g2b"
STATE_PATH = ROOT / "out" / "collect_state.json"
OUT_DIR = ROOT / "data"

WORK_CODE = {"servc": 2, "cnstwk": 1}            # 사내 데이터와 동일: 공사=1, 용역=2
# 예가 범위 분류: 15개 예비가격(bsisPlnprc)의 기초금액 대비 편차 (max-min)/2 → ±2% / ±3% 가 전체의 99.9%
RANGE_BINS = [-1.0, 1.5, 2.5, 3.5, 100.0]
RANGE_LABELS = ["±1", "±2", "±3", "±3+"]
WORK_OP = {"servc": "Servc", "cnstwk": "Cnstwk"}
KEY_COLS = ["bidNtceNo", "bidNtceOrd", "bidClsfcNo", "rbidNo"]
PAGE_SIZE = 999
SLEEP_SEC = 0.15                                   # 30 tps 한도 대비 여유


class ApiError(RuntimeError):
    pass


class DailyLimitReached(RuntimeError):
    pass


# ---------------------------------------------------------------- 상태
def load_state(path: Path = STATE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"date": "", "calls_today": 0, "tasks": {}}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(path)


def touch_day(state: dict) -> None:
    today = dt.date.today().isoformat()
    if state.get("date") != today:
        state["date"] = today
        state["calls_today"] = 0


# ---------------------------------------------------------------- 클라이언트
class G2BClient:
    def __init__(self, service_key: str, state: dict, daily_limit: int = 950, dry_run: bool = False,
                 base_url: str = BASE_URL):
        self.key = service_key
        self.base_url = base_url
        self.state = state
        self.daily_limit = daily_limit
        self.dry_run = dry_run
        self.session = requests.Session()

    def _redact(self, msg: str) -> str:
        """오류 메시지에 섞인 인증키(requests 예외의 요청 URL 등)를 ***로 가린다. 로그·예외에 키를 남기지 않는다."""
        if not self.key:
            return msg
        for secret in {self.key, quote(self.key, safe="")}:
            msg = msg.replace(secret, "***")
        return msg

    def get(self, op: str, params: dict, retries: int = 3) -> dict:
        """한 페이지를 호출해 body(dict)를 돌려준다. 일일 한도 도달 시 DailyLimitReached."""
        touch_day(self.state)
        if self.state["calls_today"] >= self.daily_limit:
            raise DailyLimitReached(f"일일 호출 한도 {self.daily_limit}건 도달 ({self.state['date']})")
        q = {"ServiceKey": self.key, "type": "json", **params}
        last_err: str | None = None
        for attempt in range(retries):
            try:
                self.state["calls_today"] += 1
                r = self.session.get(f"{self.base_url}/{op}", params=q, timeout=60)
                time.sleep(SLEEP_SEC)
                if r.status_code >= 500:
                    raise ApiError(f"HTTP {r.status_code}")
                try:
                    data = r.json()
                except ValueError:
                    raise ApiError(f"JSON 아님 (HTTP {r.status_code}): {r.text[:200]}")
                return parse_body(data)
            except (requests.ConnectionError, requests.Timeout, ApiError) as e:
                msg = self._redact(str(e))
                last_err = msg
                # 인증키/트래픽 오류는 재시도 무의미 → 즉시 중단
                if any(code in msg for code in ("resultCode=22", "resultCode=30", "resultCode=31", "resultCode=07")):
                    raise ApiError(msg) from None
                time.sleep(2 ** attempt)
        raise ApiError(self._redact(f"{op} 재시도 {retries}회 실패: {last_err}"))


def parse_body(data: dict) -> dict:
    """정상 응답은 body dict, 오류 응답(두 가지 형태)은 ApiError."""
    if "response" in data:
        header = data["response"].get("header", {})
        if header.get("resultCode") != "00":
            raise ApiError(f"resultCode={header.get('resultCode')} {header.get('resultMsg')}")
        body = data["response"].get("body") or {}
        if not isinstance(body.get("items"), list):
            body["items"] = []
        return body
    err = next((v for k, v in data.items() if "ResponseError" in k), None)
    if err:
        h = err.get("header", {})
        raise ApiError(f"resultCode={h.get('resultCode')} {h.get('resultMsg')}")
    raise ApiError(f"알 수 없는 응답 형식: {str(data)[:200]}")


# ---------------------------------------------------------------- 유틸
def month_range(start: str, end: str) -> list[str]:
    """'2024-01','2024-03' -> ['2024-01','2024-02','2024-03']"""
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    while (y, m) <= (ye, me):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def month_bounds(ym: str) -> tuple[str, str]:
    y, m = map(int, ym.split("-"))
    last = calendar.monthrange(y, m)[1]
    return f"{y:04d}{m:02d}010000", f"{y:04d}{m:02d}{last:02d}2359"


def bid_key(row: dict) -> str:
    return "|".join(str(row.get(c, "")).strip() for c in KEY_COLS)


def append_jsonl(path: Path, rows: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, **meta}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 페이지 순회
def crawl_pages(client: G2BClient, op: str, base_params: dict, task_id: str, raw_path: Path,
                meta: dict, max_pages: int | None = None) -> dict:
    """task_id 단위로 페이지를 순회하며 jsonl에 append. 상태에 next_page/complete 기록."""
    tasks = client.state.setdefault("tasks", {})
    t = tasks.setdefault(task_id, {"next_page": 1, "total_count": None, "rows": 0, "complete": False})
    if t["complete"]:
        return t
    pages_done = 0
    while True:
        page = t["next_page"]
        body = client.get(op, {**base_params, "pageNo": page, "numOfRows": PAGE_SIZE})
        items = body["items"]
        total = int(body.get("totalCount") or 0)
        t["total_count"] = total
        append_jsonl(raw_path, items, {**meta, "_page": page})
        t["rows"] += len(items)
        t["next_page"] = page + 1
        pages_done += 1
        total_pages = max(1, -(-total // PAGE_SIZE))
        print(f"  [{task_id}] page {page}/{total_pages} rows={len(items)} (누적 {t['rows']}/{total}) calls_today={client.state['calls_today']}")
        if page >= total_pages or not items:
            t["complete"] = True
        save_state(client.state)
        if t["complete"] or (max_pages and pages_done >= max_pages):
            return t


def run_market(client: G2BClient, works: list[str], months: list[str], max_pages: int | None) -> None:
    for ym in months:
        bgn, end = month_bounds(ym)
        base = {"inqryDiv": 1, "inqryBgnDt": bgn, "inqryEndDt": end}
        for w in works:
            W = WORK_OP[w]
            crawl_pages(client, f"getOpengResultListInfo{W}PreparPcDetail", base,
                        f"market:{w}:detail:{ym}", RAW_DIR / f"detail_{w}_{ym}.jsonl",
                        {"_work": w, "_ym": ym}, max_pages)
            crawl_pages(client, f"getOpengResultListInfo{W}", base,
                        f"market:{w}:list:{ym}", RAW_DIR / f"list_{w}_{ym}.jsonl",
                        {"_work": w, "_ym": ym}, max_pages)


def run_agency(client: G2BClient, works: list[str], months: list[str], agency: str, max_pages: int | None) -> None:
    for ym in months:
        bgn, end = month_bounds(ym)
        base = {"inqryDiv": 1, "inqryBgnDt": bgn, "inqryEndDt": end, "dminsttNm": agency}
        for w in works:
            W = WORK_OP[w]
            list_path = RAW_DIR / f"agency_{w}_{ym}.jsonl"
            crawl_pages(client, f"getScsbidListSttus{W}PPSSrch", base,
                        f"agency:{w}:list:{ym}", list_path, {"_work": w, "_ym": ym, "_agency": agency}, max_pages)
            # 공고번호별 예비가격상세 (inqryDiv=2. 2026-11-01 실측: 4 는 resultCode 07 '입력범위값 초과')
            notices = sorted({json.loads(l)["bidNtceNo"] for l in list_path.read_text().splitlines()}) if list_path.exists() else []
            t = client.state["tasks"].setdefault(f"agency:{w}:detail:{ym}", {"done": [], "complete": False})
            for no in notices:
                if no in t["done"]:
                    continue
                body = client.get(f"getOpengResultListInfo{W}PreparPcDetail", {"inqryDiv": 2, "bidNtceNo": no, "pageNo": 1, "numOfRows": PAGE_SIZE})
                append_jsonl(RAW_DIR / f"detail_{w}_{ym}.jsonl", body["items"], {"_work": w, "_ym": ym, "_agency": agency})
                t["done"].append(no)
                save_state(client.state)
                print(f"  [agency:{w}:detail:{ym}] {no} rows={len(body['items'])} calls_today={client.state['calls_today']}")
            t["complete"] = True
            save_state(client.state)


# ---------------------------------------------------------------- parquet 빌드
def _read_jsonl(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return pd.DataFrame(rows)


def build_detail(detail: pd.DataFrame, listing: pd.DataFrame | None) -> pd.DataFrame:
    """예비가격상세(복수예가 15행/건) → 입찰 1건 1행. 개찰결과 목록과 조인."""
    if detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    for c in KEY_COLS:
        d[c] = d[c].astype(str).str.strip()
    d["bid_key"] = d[KEY_COLS].agg("|".join, axis=1)
    d["base_amt"] = pd.to_numeric(d["bssamt"], errors="coerce")
    d["yega_amt"] = pd.to_numeric(d["plnprc"], errors="coerce")
    d = d[(d["base_amt"] > 0) & (d["yega_amt"] > 0)]
    # 입찰별 예가 범위: 15행의 예비가격(bsisPlnprc) 편차 범위. 시장 모델의 핵심 피처(±2%: std 0.50, ±3%: std 0.74).
    if "bsisPlnprc" in d.columns:
        dev = (pd.to_numeric(d["bsisPlnprc"], errors="coerce") / d["base_amt"] - 1.0) * 100.0
        rng = dev.groupby(d["bid_key"]).agg(["min", "max", "count"])
        rng["half_range"] = (rng["max"] - rng["min"]) / 2.0
        rng.loc[rng["count"] < 2, "half_range"] = float("nan")
        rng["range_cls"] = pd.cut(rng["half_range"], RANGE_BINS, labels=RANGE_LABELS).astype(object)
        rng = rng[["half_range", "range_cls", "count"]].rename(columns={"count": "n_prices"})
    else:
        rng = None
    d = d.sort_values(["bid_key", "_page"]).drop_duplicates("bid_key", keep="first")
    out = pd.DataFrame({
        "bid_key": d["bid_key"].values,
        "bidNtceNo": d["bidNtceNo"].values,
        "bid_name": d["bidNtceNm"].values,
        "work_type": d["_work"].map(WORK_CODE).values,
        "work_type_txt": d["_work"].map({"servc": "용역", "cnstwk": "공사"}).values,
        "ym": d["_ym"].values,
        "base_amt": d["base_amt"].values,
        "yega_amt": d["yega_amt"].values,
        "tot_rsrv_num": pd.to_numeric(d["totRsrvtnPrceNum"], errors="coerce").values,
        "rsrv_upper_num": pd.to_numeric(d.get("bssamtBssUpNum"), errors="coerce").values,
        "rl_open_dt": pd.to_datetime(d["rlOpengDt"], errors="coerce").values,
    })
    out["yega_rate"] = out["yega_amt"] / out["base_amt"] * 100.0
    if rng is not None:
        out = out.merge(rng, left_on="bid_key", right_index=True, how="left")
    else:
        out["half_range"] = float("nan"); out["range_cls"] = None; out["n_prices"] = float("nan")

    if listing is not None and not listing.empty:
        l = listing.copy()
        for c in KEY_COLS:
            l[c] = l[c].astype(str).str.strip()
        l["bid_key"] = l[KEY_COLS].agg("|".join, axis=1)
        l = l.drop_duplicates("bid_key", keep="last")
        l = pd.DataFrame({
            "bid_key": l["bid_key"].values,
            "open_dt": pd.to_datetime(l.get("opengDt", l.get("rlOpengDt")), errors="coerce").values,
            "bidder_count": pd.to_numeric(l["prtcptCnum"], errors="coerce").values,
            "dminsttNm": l["dminsttNm"].values,
            "progress": l["progrsDivCdNm"].values if "progrsDivCdNm" in l else None,
        })
        out = out.merge(l, on="bid_key", how="left")
    else:
        out["open_dt"] = pd.NaT
        out["bidder_count"] = float("nan")
        out["dminsttNm"] = None
        out["progress"] = None

    out["open_dt"] = out["open_dt"].fillna(out["rl_open_dt"])
    out["source"] = "g2b"
    return out.sort_values(["open_dt", "bid_key"]).reset_index(drop=True)


def build(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    frames = []
    for w in WORK_CODE:
        dpaths = sorted(RAW_DIR.glob(f"detail_{w}_*.jsonl"))
        lpaths = sorted(RAW_DIR.glob(f"list_{w}_*.jsonl")) + sorted(RAW_DIR.glob(f"agency_{w}_*.jsonl"))
        if not dpaths:
            continue
        df = build_detail(_read_jsonl(dpaths), _read_jsonl(lpaths) if lpaths else None)
        if not df.empty:
            df.to_parquet(out_dir / f"g2b_detail_{w}.parquet", index=False)
            frames.append(df)
    if not frames:
        print("빌드할 원시 데이터가 없습니다.")
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(out_dir / "g2b_detail_all.parquet", index=False)
    print_build_report(all_df)
    return all_df


def print_build_report(df: pd.DataFrame) -> None:
    print("=== 나라장터 수집 데이터 빌드 결과 ===")
    print(f"복수예가 입찰 건수(기초·예정가격 모두 있음): {len(df):,}")
    inr = df["yega_rate"].between(96.5, 103.5)
    print(f"사정률 96.5~103.5% 이내: {int(inr.sum()):,} ({inr.mean()*100:.1f}%)")
    r = df.loc[inr, "yega_rate"]
    print(f"사정률(%) mean {r.mean():.4f} | std {r.std(ddof=0):.4f} | Q25 {r.quantile(.25):.3f} | Q50 {r.quantile(.5):.3f} | Q75 {r.quantile(.75):.3f}")
    print("공사/용역별:", df.groupby("work_type_txt")["yega_rate"].agg(["size", "mean"]).round(4).to_dict("index"))
    print(f"참여업체수 조인 성공: {df['bidder_count'].notna().mean()*100:.1f}% | 중앙값 {df['bidder_count'].median()}")
    if "range_cls" in df:
        print("예가 범위 분포:", df["range_cls"].value_counts(dropna=False).to_dict())
        print("범위별 사정률 std:", df.groupby("range_cls")["yega_rate"].std(ddof=0).round(4).to_dict())
    print("월별 건수:", df.groupby("ym").size().to_dict())


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="나라장터 낙찰정보 수집기")
    ap.add_argument("--mode", choices=["market", "agency"], default="market")
    ap.add_argument("--work", default="servc,cnstwk", help="servc,cnstwk 중 선택 (쉼표 구분)")
    ap.add_argument("--start", help="YYYY-MM")
    ap.add_argument("--end", help="YYYY-MM (기본: start)")
    ap.add_argument("--agency", default="", help="agency 모드 수요기관명 (예: 한국남동발전)")
    ap.add_argument("--daily-limit", type=int, default=950)
    ap.add_argument("--max-pages", type=int, default=None, help="태스크당 이번 실행에서 처리할 최대 페이지 (테스트용)")
    ap.add_argument("--build", action="store_true", help="수집 없이 원시 jsonl → parquet 빌드만")
    ap.add_argument("--no-build", action="store_true", help="수집 후 빌드 생략")
    ap.add_argument("--reset-state", action="store_true")
    args = ap.parse_args(argv)

    if args.build:
        build()
        return 0
    if not args.start:
        ap.error("--start 가 필요합니다")
    if args.mode == "agency" and not args.agency:
        ap.error("agency 모드는 --agency 가 필요합니다")

    load_dotenv(ROOT / ".env")
    key = os.environ.get("G2B_SERVICE_KEY", "")
    if not key:
        print(".env 에 G2B_SERVICE_KEY 가 없습니다", file=sys.stderr)
        return 1

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()
    state = load_state()
    touch_day(state)
    client = G2BClient(key, state, daily_limit=args.daily_limit)
    works = [w.strip() for w in args.work.split(",") if w.strip()]
    months = month_range(args.start, args.end or args.start)
    print(f"모드={args.mode} 업무={works} 기간={months[0]}~{months[-1]} ({len(months)}개월) 오늘 호출={state['calls_today']}/{args.daily_limit}")

    rc = 0
    try:
        if args.mode == "market":
            run_market(client, works, months, args.max_pages)
        else:
            run_agency(client, works, months, args.agency, args.max_pages)
        print("수집 완료.")
    except DailyLimitReached as e:
        print(f"\n[중단] {e}. 내일 같은 명령으로 재실행하면 이어서 수집합니다.")
        rc = 2
    except ApiError as e:
        print(f"\n[오류] {e}. 상태는 저장되었으며 재실행 시 이어서 수집합니다.", file=sys.stderr)
        rc = 3
    except KeyboardInterrupt:
        print("\n[중단] 사용자 중단. 상태 저장됨.")
        rc = 130
    finally:
        save_state(state)

    if not args.no_build:
        build()
    return rc


if __name__ == "__main__":
    sys.exit(main())
