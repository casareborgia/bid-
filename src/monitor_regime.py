"""예가 규정 변경 감시 스크립트 (Phase 7).

시장 모델은 나라장터 복수예가 입찰의 사정률 분포가 예가 범위(±2%/±3%)별로 상수라는 전제 위에 있다.
이 전제가 깨지는지(예가 규정·추첨 방식 변경) 연 1회 최근 1개월만 수집해 점검한다.

사용 예:
  uv run python src/monitor_regime.py                                    # 지난달 수집 후 점검
  uv run python src/monitor_regime.py --month 2026-09 --skip-collect     # 기존 데이터로 2026-09 점검
  uv run python src/monitor_regime.py --month 2026-09 --skip-collect --no-build  # 빌드도 생략
  uv run python src/monitor_regime.py --json                             # JSON 출력
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_g2b_api as cg  # noqa: E402
import evaluate as E  # noqa: E402
import features as F  # noqa: E402
import train_model as T  # noqa: E402

REFERENCE = {"±2": {"mean": 99.987, "std": 0.509}, "±3": {"mean": 99.855, "std": 0.740}}  # 기준 분포 (538,907건, 2026-10-31)
TOL_MEAN = 0.10  # %p
TOL_STD = 0.05
MIN_N = 500  # 범위별 표본이 이보다 적으면 판정 보류(경고만)
COVERAGE_RANGE = (0.45, 0.55)
VALID_Q75_RANGE = (0.72, 0.78)  # 마진 0 기준 Q75 유효율
OUT_DIR = ROOT / "out"


def previous_complete_month(today: dt.date | None = None) -> str:
    """오늘 기준 지난달 YYYY-MM 을 반환한다 (1월이면 전년 12월)."""
    if today is None:
        today = dt.date.today()
    y, m = today.year, today.month
    if m == 1:
        return f"{y - 1:04d}-12"
    return f"{y:04d}-{m - 1:02d}"


def collect_month(
    ym: str,
    works: tuple[str, ...] = ("servc", "cnstwk"),
    client: cg.G2BClient | None = None,
    daily_limit: int = 950,
) -> dict:
    """지정한 달 1개월만 수집한다. 일일 한도 도달 시 complete=False 반환."""
    if client is None:
        load_dotenv(ROOT / ".env")
        key = os.environ.get("G2B_SERVICE_KEY")
        if not key:
            raise RuntimeError(".env 에 G2B_SERVICE_KEY 가 없습니다")
        state = cg.load_state()
        cg.touch_day(state)
        client = cg.G2BClient(key, state, daily_limit=daily_limit)

    try:
        cg.run_market(client, list(works), [ym], None)
    except cg.DailyLimitReached as e:
        cg.save_state(client.state)
        return {
            "ym": ym,
            "complete": False,
            "reason": str(e),
            "calls_today": client.state.get("calls_today", 0),
        }

    cg.save_state(client.state)
    return {
        "ym": ym,
        "complete": True,
        "reason": None,
        "calls_today": client.state.get("calls_today", 0),
    }


def range_stats_for_month(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    """train_model.load_market() 의 결과에서 ym 행만 골라 range_cls 별 n, mean, std(ddof=0) 계산."""
    if df.empty or "contract_date" not in df.columns:
        return pd.DataFrame(columns=["n", "mean", "std"])

    dt_series = pd.to_datetime(df["contract_date"], errors="coerce")
    mask = dt_series.dt.strftime("%Y-%m") == ym
    sub = df[mask]
    if sub.empty:
        return pd.DataFrame(columns=["n", "mean", "std"])

    g = sub.groupby("range_cls", observed=True)["yega_rate"]
    res = pd.DataFrame({
        "n": g.size().astype(int),
        "mean": g.mean().astype(float),
        "std": g.std(ddof=0).fillna(0.0).astype(float),
    })
    res.index = res.index.astype(str)
    return res


def check_drift(
    stats: pd.DataFrame,
    reference: dict | None = None,
    tol_mean: float = TOL_MEAN,
    tol_std: float = TOL_STD,
    min_n: int | None = None,
) -> tuple[list[str], list[str]]:
    """기준 분포 대비 드리프트(평균/표준편차)와 표본 부족 경고를 판정한다."""
    if reference is None:
        reference = REFERENCE
    if min_n is None:
        min_n = MIN_N
    drift_msgs: list[str] = []
    warnings: list[str] = []

    for rc in sorted(reference.keys()):
        ref = reference[rc]
        ref_mean = ref["mean"]
        ref_std = ref["std"]

        if stats.empty or rc not in stats.index:
            warnings.append(f"{rc}: 표본 0건 < {min_n}, 판정 보류")
            continue

        n = int(stats.loc[rc, "n"])
        if n < min_n:
            warnings.append(f"{rc}: 표본 {n}건 < {min_n}, 판정 보류")
            continue

        mean = float(stats.loc[rc, "mean"])
        std = float(stats.loc[rc, "std"])

        diff_mean = mean - ref_mean
        if abs(diff_mean) > tol_mean:
            drift_msgs.append(
                f"{rc}: 평균 {mean:.3f} (기준 {ref_mean:.3f}, 차 {diff_mean:+.3f} > {tol_mean:.2f})"
            )

        diff_std = std - ref_std
        if abs(diff_std) > tol_std:
            drift_msgs.append(
                f"{rc}: 표준편차 {std:.3f} (기준 {ref_std:.3f}, 차 {diff_std:+.3f} > {tol_std:.2f})"
            )

    return drift_msgs, warnings


def backtest_month(ym: str) -> dict:
    """해당 월 홀드아웃 백테스트를 수행한다."""
    return E.backtest("market", [ym], sample=5)


def judge_backtest(bt: dict) -> list[str]:
    """백테스트 지표가 목표 범위를 벗어나는지 확인한다."""
    msgs: list[str] = []
    overall = bt.get("overall", {})
    cov = overall.get("coverage_q25_q75")
    if cov is not None and not (COVERAGE_RANGE[0] <= cov <= COVERAGE_RANGE[1]):
        msgs.append(
            f"커버리지 {cov * 100:.1f}% (목표 {int(COVERAGE_RANGE[0]*100)}~{int(COVERAGE_RANGE[1]*100)}%)"
        )
    val_q75 = overall.get("valid_q75_nomargin")
    if val_q75 is not None and not (VALID_Q75_RANGE[0] <= val_q75 <= VALID_Q75_RANGE[1]):
        msgs.append(
            f"Q75 유효율 {val_q75 * 100:.1f}% (목표 {int(VALID_Q75_RANGE[0]*100)}~{int(VALID_Q75_RANGE[1]*100)}%)"
        )
    return msgs


def write_report(
    ym: str,
    collect_info: dict | None,
    stats: pd.DataFrame,
    drift: list[str],
    warnings: list[str],
    bt: dict | None,
    bt_msgs: list[str],
    path: Path,
) -> str:
    path = Path(path)
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if collect_info is None:
        collect_str = "기존 데이터 사용"
    elif not collect_info.get("complete", False):
        collect_str = "수집 미완료(일일 한도) — 내일 같은 명령으로 재실행하면 이어받음"
    else:
        collect_str = f"수집 완료 (오늘 호출 {collect_info.get('calls_today', 0)}건)"

    lines = [
        f"# 예가 규정 변경 감시 — {ym}",
        "",
        f"- 생성 시각: {now_str}",
        f"- 수집 정보: {collect_str}",
        "",
        "## 예가 범위별 사정률 통계",
        "",
        "| 범위 | n | mean | std | 기준 mean | 기준 std | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]

    all_ranges = sorted(set(list(REFERENCE.keys()) + list(stats.index if not stats.empty else [])))
    for r in all_ranges:
        ref = REFERENCE.get(r, {})
        ref_mean = f"{ref['mean']:.3f}" if "mean" in ref else "-"
        ref_std = f"{ref['std']:.3f}" if "std" in ref else "-"
        if not stats.empty and r in stats.index:
            n_val = int(stats.loc[r, "n"])
            mean_val = f"{float(stats.loc[r, 'mean']):.3f}"
            std_val = f"{float(stats.loc[r, 'std']):.3f}"
            r_drifts = [m for m in drift if m.startswith(f"{r}:")]
            r_warns = [m for m in warnings if m.startswith(f"{r}:")]
            if r_drifts:
                verdict = "변화 감지"
            elif r_warns:
                verdict = "보류"
            else:
                verdict = "정상"
        else:
            n_val = 0
            mean_val = "-"
            std_val = "-"
            verdict = "보류"
        lines.append(f"| {r} | {n_val:,} | {mean_val} | {std_val} | {ref_mean} | {ref_std} | {verdict} |")

    lines.append("")
    lines.append("## 백테스트 요약")
    lines.append("")

    if bt is None:
        lines.append("- 백테스트 생략/실패")
        for m in bt_msgs:
            lines.append(f"  - {m}")
    else:
        o = bt.get("overall", {})
        n = o.get("n", 0)
        cov = o.get("coverage_q25_q75", 0.0)
        vq25 = o.get("valid_q25_nomargin", 0.0)
        vq50 = o.get("valid_q50_nomargin", 0.0)
        vq75 = o.get("valid_q75_nomargin", 0.0)
        ml_mae = o.get("ml_mae", 0.0)
        base_mae = o.get("baseline_mae", 0.0)

        lines.append("| n | Q25~Q75 커버리지 | 유효율 Q25 (마진 0) | 유효율 Q50 (마진 0) | 유효율 Q75 (마진 0) | ML MAE | Baseline MAE |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(f"| {n:,} | {cov*100:.1f}% | {vq25*100:.1f}% | {vq50*100:.1f}% | {vq75*100:.1f}% | {ml_mae:.4f} | {base_mae:.4f} |")
        if bt_msgs:
            lines.append("")
            for m in bt_msgs:
                lines.append(f"- {m}")

    lines.append("")
    lines.append("## 판정")
    lines.append("")

    change_msgs = list(drift) + [m for m in bt_msgs if not m.startswith("백테스트 불가")]
    if change_msgs:
        lines.append("- **변화 감지**:")
        for m in change_msgs:
            lines.append(f"  - {m}")
        lines.append("- 조치: 최근 월을 포함해 재빌드(--build)·재학습(train_model.py --source market) 후 CLAUDE.md 기준 수치를 갱신한다.")
    else:
        lines.append("- **이상 없음**: 범위별 분포와 홀드아웃 성적이 기준 안이다. 모델 갱신 불필요.")

    if warnings:
        lines.append("")
        lines.append("## 경고")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="예가 규정 변경 감시 스크립트")
    parser.add_argument("--month", default=previous_complete_month(), help="점검 대상 월 (YYYY-MM, 기본 지난달)")
    parser.add_argument("--skip-collect", action="store_true", help="수집 생략 (기존 parquet 사용)")
    parser.add_argument("--no-build", action="store_true", help="빌드 생략")
    parser.add_argument("--works", default="servc,cnstwk", help="수집 업무 구분 (쉼표 구분)")
    parser.add_argument("--daily-limit", type=int, default=950, help="일일 API 호출 한도")
    parser.add_argument("--out", type=Path, default=None, help="리포트 출력 파일 경로")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 결과 출력")

    args = parser.parse_args(argv)
    ym = args.month
    out_path = args.out or (OUT_DIR / f"regime_check_{ym}.md")

    collect_info = None
    if not args.skip_collect:
        works = tuple(w.strip() for w in args.works.split(",") if w.strip())
        try:
            collect_info = collect_month(ym, works=works, daily_limit=args.daily_limit)
        except cg.ApiError as e:
            sys.stderr.write(f"API 오류: {e}\n")
            return 4
        except RuntimeError as e:
            sys.stderr.write(f"런타임 오류: {e}\n")
            return 4

        if not collect_info["complete"]:
            stats = pd.DataFrame(columns=["n", "mean", "std"])
            warn_msg = f"수집 미완료: {collect_info.get('reason')}"
            rep_text = write_report(ym, collect_info, stats, [], [warn_msg], None, [], out_path)
            if args.json:
                res = {
                    "exit_code": 2,
                    "ym": ym,
                    "collect_info": collect_info,
                    "drift": [],
                    "warnings": [warn_msg],
                    "bt_msgs": [],
                    "stats": stats.to_dict(orient="index"),
                }
                print(json.dumps(res, ensure_ascii=False, indent=2))
            else:
                print(rep_text)
            return 2

    if not args.no_build:
        cg.build()

    df, meta = T.load_market()
    stats = range_stats_for_month(df, ym)
    if stats.empty or stats["n"].sum() == 0:
        sys.stderr.write("해당 월 데이터 없음\n")
        warn_msg = f"{ym} 데이터 없음"
        rep_text = write_report(ym, collect_info, stats, [], [warn_msg], None, [], out_path)
        if args.json:
            res = {
                "exit_code": 5,
                "ym": ym,
                "collect_info": collect_info,
                "drift": [],
                "warnings": [warn_msg],
                "bt_msgs": [],
                "stats": stats.to_dict(orient="index"),
            }
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(rep_text)
        return 5

    drift, warnings = check_drift(stats, reference=REFERENCE, min_n=MIN_N)

    try:
        bt = backtest_month(ym)
        bt_msgs = judge_backtest(bt)
    except ValueError as e:
        bt = None
        bt_msgs = [f"백테스트 불가: {e}"]

    rep_text = write_report(ym, collect_info, stats, drift, warnings, bt, bt_msgs, out_path)

    judge_msgs = judge_backtest(bt) if bt is not None else []
    exit_code = 3 if (drift or judge_msgs) else 0

    if args.json:
        stats_dict = {
            str(k): {"n": int(v["n"]), "mean": float(v["mean"]), "std": float(v["std"])}
            for k, v in stats.to_dict(orient="index").items()
        }
        res = {
            "exit_code": exit_code,
            "ym": ym,
            "collect_info": collect_info,
            "drift": drift,
            "warnings": warnings,
            "bt_msgs": bt_msgs,
            "stats": stats_dict,
        }
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(rep_text)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
