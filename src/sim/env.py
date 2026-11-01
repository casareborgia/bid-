"""입찰 환경 시뮬레이터 (시뮬레이터 1단계).

나라장터 복수예가 절차를 2026-10 개찰 12,601건의 예비가격상세로 확인한 규칙대로 재현한다.
  예비가격 15개 생성
    ±2%: [-2, +2]% 를 15등분하고 칸마다 1개씩 균등 난수 (실측 92%가 이 구조)
    ±3%: 아래 [-3, 0)% 를 8등분, 위 [0, +3]% 를 7등분하고 칸마다 1개씩 (실측 99%)
  추첨
    입찰자마다 15개 번호 중 서로 다른 2개를 고른다. 번호 선택은 가격·번호와 무관하게 균등하다
    (실측: 번호별 선택 비율 0.998~1.002, 선택 순위와 가격 순위 상관 -0.014).
    가장 많이 선택된 4개(동률은 무작위)의 평균이 예정가격이다 (실측: 예정가격 = 추첨 4개 평균 100%, 상위 4개 = 추첨 98.6%).
  보정(실측과의 차이):
    ±3% 는 추첨이 낮은 가격 쪽으로 약간 치우친다(가격 순위 1위 1.07배 ~ 15위 0.92배, 사정률 평균 −0.05%p).
    `draw_bias`(가격 순위 가중 β, 기본 ±2 0 / ±3 0.05)로 평균을 맞춘다. ±2% 는 치우침이 없다.
    남는 차이: ±3% 표준편차가 실측 0.740 보다 약 0.027 크다(같은 15개에서 무작위 4개를 뽑아도 0.752 → 생성·추첨 양쪽에
    작은 상관이 있다). 분포를 정확히 맞춰야 하면 `EmpiricalRates`(실측 사정률 재표본)를 쓴다.
  낙찰자 결정(적격심사 단순화): 낙찰하한금액 이상 유효 투찰 중 최저가가 1순위.
    낙찰하한금액 = (예정가격 - A) × 하한율 + A (공사만 A값).

모든 함수는 numpy Generator 를 받아 재현 가능하다. 금액 단위는 원, 사정률·편차는 %.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_PRICES = 15
N_DRAWN = 4
PICKS_PER_BIDDER = 2
DRAW_BIAS = {2: 0.0, 3: 0.05}   # 가격 순위 가중 β: 가중치 ∝ exp(-β·(순위-7)/7), 순위 0=최저가
# 예가 범위별 칸 구성: (구간 시작 %, 구간 끝 %, 칸 수)
LAYOUT = {
    2: [(-2.0, 2.0, 15)],
    3: [(-3.0, 0.0, 8), (0.0, 3.0, 7)],
}


def reserve_deviations(range_pct: int, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """예비가격 15개의 기초금액 대비 편차(%). size 를 주면 (size, 15) 배열."""
    if range_pct not in LAYOUT:
        raise ValueError(f"예가 범위는 2 또는 3 이어야 합니다: {range_pct}")
    shape = (size,) if size is not None else ()
    parts = []
    for lo, hi, k in LAYOUT[range_pct]:
        edges = np.linspace(lo, hi, k + 1)
        parts.append(edges[:-1] + (edges[1:] - edges[:-1]) * rng.random(shape + (k,)))
    dev = np.concatenate(parts, axis=-1)
    return dev  # 번호는 가격과 무관하므로 정렬된 상태로 두어도 추첨 결과 분포는 같다


def draw_numbers(n_bidders: int, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """입찰자들이 2개씩 고른 번호의 선택 횟수로 상위 4개 번호를 뽑는다. 반환은 (…, 4) 번호 인덱스(0~14).
    n_bidders 가 0 이면 무작위 4개(실제로는 유찰이지만 분포 계산용)."""
    shape = (size,) if size is not None else ()
    m = int(np.prod(shape)) if shape else 1
    counts = np.zeros((m, N_PRICES))
    for _ in range(max(n_bidders, 0)):
        # 서로 다른 2개: 무작위 순열 앞 2개와 같다 — 15개 균등 난수의 argpartition
        picks = np.argpartition(rng.random((m, N_PRICES)), PICKS_PER_BIDDER, axis=1)[:, :PICKS_PER_BIDDER]
        np.add.at(counts, (np.arange(m)[:, None], picks), 1)
    score = counts + rng.random((m, N_PRICES)) * 0.5          # 동률은 무작위로 깬다(정수 횟수 차이보다 작은 잡음)
    top = np.argpartition(-score, N_DRAWN, axis=1)[:, :N_DRAWN]
    return top.reshape(shape + (N_DRAWN,))


def simulate_yega_rates(range_pct: int, n: int, rng: np.random.Generator, n_bidders: int | None = None,
                        draw_bias: float | None = None) -> np.ndarray:
    """사정률(예정가격/기초금액×100) n 개.
    n_bidders=None: 번호 선택이 가격과 무관하다는 실측에 따라 4개를 바로 추출하되, 가격 순위 가중(draw_bias, 기본 DRAW_BIAS)을
    Gumbel-top-k 로 준다. n_bidders 를 주면 입찰자별 번호 선택을 그대로 흉내 낸다(치우침 없음, 느림)."""
    dev = np.sort(reserve_deviations(range_pct, rng, size=n), axis=1)       # 열 = 가격 순위
    if n_bidders is None:
        beta = DRAW_BIAS[range_pct] if draw_bias is None else draw_bias
        logw = -beta * (np.arange(N_PRICES) - (N_PRICES - 1) / 2) / ((N_PRICES - 1) / 2)
        gumbel = logw - np.log(-np.log(rng.random((n, N_PRICES))))
        idx = np.argpartition(-gumbel, N_DRAWN, axis=1)[:, :N_DRAWN]
    else:
        idx = draw_numbers(n_bidders, rng, size=n)
    return 100.0 + np.take_along_axis(dev, idx, axis=1).mean(axis=1)


def lower_limit_amount(yega: np.ndarray | float, lower_limit_pct: float, a_value: float = 0.0):
    """낙찰하한금액 = (예정가격 - A) × 하한율 + A. predict_cli.lower_limit_amount 와 같은 식(배열 지원)."""
    return (np.asarray(yega) - a_value) * lower_limit_pct / 100.0 + a_value


@dataclass
class AuctionResult:
    yega: float
    lower_limit: float
    winner: int | None          # bids 배열의 인덱스, 유효 투찰이 없으면 None
    valid: np.ndarray           # 각 투찰의 하한 이상 여부


def run_auction(base_amt: float, range_pct: int, bids: np.ndarray, lower_limit_pct: float, rng: np.random.Generator,
                a_value: float = 0.0) -> AuctionResult:
    """한 건의 입찰을 끝까지 진행한다: 예비가격 생성 → 입찰자 수만큼 번호 선택 → 예정가격 → 하한 → 1순위."""
    bids = np.asarray(bids, dtype=float)
    dev = reserve_deviations(range_pct, rng)
    idx = draw_numbers(len(bids), rng)
    prices = np.floor(base_amt * (1 + dev / 100.0))                 # 예비가격은 원 단위
    yega = float(np.floor(prices[idx].mean()))
    lower = float(lower_limit_amount(yega, lower_limit_pct, a_value))
    valid = bids >= lower
    winner = int(np.flatnonzero(valid)[np.argmin(bids[valid])]) if valid.any() else None
    return AuctionResult(yega=yega, lower_limit=lower, winner=winner, valid=valid)


class EmpiricalRates:
    """실측 사정률 재표본기. 시장 데이터(g2b_detail_all.parquet)의 예가 범위별 사정률을 그대로 뽑는다."""

    def __init__(self, rates_by_range: dict[int, np.ndarray]):
        self.rates = {int(k): np.asarray(v, dtype=float) for k, v in rates_by_range.items()}

    @classmethod
    def from_parquet(cls, path, lo: float = 96.5, hi: float = 103.5) -> "EmpiricalRates":
        import pandas as pd
        g = pd.read_parquet(path, columns=["range_cls", "yega_rate"])
        g = g[g["yega_rate"].between(lo, hi)]
        return cls({rc: g.loc[g["range_cls"] == f"±{rc}", "yega_rate"].to_numpy() for rc in (2, 3)})

    def sample(self, range_pct: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.rates[int(range_pct)], size=n, replace=True)
