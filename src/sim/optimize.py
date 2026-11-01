"""낙찰확률 최대화 — 몬테카를로 정답 (시뮬레이터 2단계).

투찰가를 "그 금액이 정확히 낙찰하한이 되는 사정률"(환산 사정률 r)로 바꾸면 적격심사 1순위 조건이 1차원 문제가 된다.
  낙찰하한 = (기초금액 × y/100 − A) × 하한율/100 + A  (y = 실제 사정률)
  투찰가 b 의 환산 사정률 r(b) = ((b − A) × 100/하한율 + A) / 기초금액 × 100
  b ≥ 하한  ⇔  r ≥ y.  1순위 = 유효 투찰 중 최저가 = r ≥ y 인 투찰 중 r 이 가장 작은 것.
따라서 내 r 로 1순위가 될 조건은 "r ≥ y 이고, 경쟁사 r_j 중 [y, r) 에 든 것이 없다".

경쟁사 r_j 를 서로 독립·동일 분포 F 로 보면 (N = 경쟁사 수)
  P(낙찰 | r) = E_y[ 1{r ≥ y} · (1 − (F(r) − F(y)))^N ]
y 는 예정가격 분포(시장 모델 분위수가 아니라 표본: 실측 재표본 또는 env 시뮬레이션), F 는 실측 경쟁사 환산 사정률의 경험 분포.
같은 금액 동률(추첨)과 적격심사 점수 미달은 무시한다.
"""
from __future__ import annotations

import numpy as np


def implied_rate(bid: np.ndarray | float, base_amt: np.ndarray | float, lower_limit_pct: np.ndarray | float,
                 a_value: np.ndarray | float = 0.0):
    """투찰가 → 환산 사정률(%). 이 사정률에서 낙찰하한이 정확히 투찰가와 같다."""
    bid, base_amt, ll, a = (np.asarray(x, dtype=float) for x in (bid, base_amt, lower_limit_pct, a_value))
    return ((bid - a) * 100.0 / ll + a) / base_amt * 100.0


def bid_from_rate(rate: np.ndarray | float, base_amt: float, lower_limit_pct: float, a_value: float = 0.0, unit: int = 1):
    """환산 사정률 → 투찰가(원). unit 단위로 올림(하한 미달을 피하는 쪽)."""
    lower = (base_amt * np.asarray(rate, dtype=float) / 100.0 - a_value) * lower_limit_pct / 100.0 + a_value
    return np.ceil(lower / unit) * unit


def win_prob_curve(r_grid: np.ndarray, y_samples: np.ndarray, competitor_rates: np.ndarray, n_competitors: int,
                   chunk: int = 4096) -> np.ndarray:
    """각 r 의 낙찰확률. competitor_rates 는 경쟁사 환산 사정률 표본(경험 분포 F), y_samples 는 실제 사정률 표본."""
    comp = np.sort(np.asarray(competitor_rates, dtype=float))
    F = lambda x: np.searchsorted(comp, x, side="left") / len(comp)    # noqa: E731  P(r_j < x)
    r = np.asarray(r_grid, dtype=float)
    Fr = F(r)
    out = np.zeros(len(r))
    y = np.asarray(y_samples, dtype=float)
    for i in range(0, len(y), chunk):
        yc = y[i:i + chunk]
        Fy = F(yc)
        p_gap = np.clip(1.0 - (Fr[None, :] - Fy[:, None]), 0.0, 1.0) ** n_competitors
        out += np.where(r[None, :] >= yc[:, None], p_gap, 0.0).sum(axis=0)
    return out / len(y)


def optimal_rate(y_samples: np.ndarray, competitor_rates: np.ndarray, n_competitors: int,
                 lo: float = 97.0, hi: float = 103.0, step: float = 0.005) -> dict:
    """낙찰확률이 최대인 환산 사정률과 곡선."""
    grid = np.round(np.arange(lo, hi + step / 2, step), 6)
    p = win_prob_curve(grid, y_samples, competitor_rates, n_competitors)
    k = int(np.argmax(p))
    return {"rate": float(grid[k]), "win_prob": float(p[k]), "grid": grid, "curve": p,
            "valid_prob": float((np.asarray(y_samples) <= grid[k]).mean())}


def replay_win(our_rate: float, y: float, competitor_rates: np.ndarray) -> bool:
    """과거 입찰 1건 재연: 내 환산 사정률을 끼워 넣었을 때 1순위인가(실제 y·실제 경쟁사 투찰 그대로)."""
    c = np.asarray(competitor_rates, dtype=float)
    return bool(our_rate >= y and not ((c >= y) & (c < our_rate)).any())


def win_prob_curve_bootstrap(r_grid: np.ndarray, y_samples: np.ndarray, auctions_rates: list[np.ndarray]) -> np.ndarray:
    """경쟁사를 독립 분포 대신 '과거 입찰의 경쟁사 투찰 묶음'으로 재표본해 P(낙찰|r) 을 계산한다.
    입찰 안의 상관(같은 입찰 참가자들이 함께 높거나 낮게 쓰는 경향)과 참가자 수를 그대로 반영한다.
    과거 입찰 a 와 실제 사정률 y 쌍마다, 내가 이기는 r 의 구간은 [y, y 이상인 첫 경쟁사 투찰) 이다."""
    grid = np.asarray(r_grid, dtype=float)
    y = np.asarray(y_samples, dtype=float)
    acc = np.zeros(len(grid) + 1)
    for c in auctions_rates:
        c = np.sort(np.asarray(c, dtype=float))
        k = np.searchsorted(c, y, side="left")                      # y 이상인 첫 경쟁사
        nxt = np.where(k < len(c), c[np.minimum(k, len(c) - 1)], np.inf)
        lo = np.searchsorted(grid, y, side="left")                   # r ≥ y 인 첫 격자
        hi = np.searchsorted(grid, nxt, side="left")                 # r < nxt 인 마지막 격자 다음
        np.add.at(acc, lo, 1.0)
        np.add.at(acc, hi, -1.0)
    return np.cumsum(acc)[:-1] / (len(y) * len(auctions_rates))
