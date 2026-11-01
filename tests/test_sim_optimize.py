"""낙찰확률 최대화(sim.optimize) 테스트: 환산 사정률 역변환, 해석식 vs 무작위 시뮬레이션, 재연 판정."""
import numpy as np
import pytest

from sim import optimize as O


def test_implied_rate_roundtrip_with_a_value():
    base, ll, a = 184_521_490, 89.745, 11_684_881
    for rate in (99.0, 100.153, 101.2):
        bid = O.bid_from_rate(rate, base, ll, a)
        r = O.implied_rate(bid, base, ll, a)
        assert r >= rate and r - rate < 1e-6                           # 원 단위 올림 → 하한 바로 위
    # 실제 사례: 2026-10 서울시 전기공사 1순위 167,052,282원, 실제 사정률 100.153% → 환산 사정률이 바로 위
    r_win = O.implied_rate(167_052_282, base, ll, a)
    assert 100.153 < r_win < 100.155


def test_win_prob_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 0.5, 20_000)
    comp = rng.normal(99.9, 0.6, 50_000)
    N = 10
    grid = np.array([99.5, 100.0, 100.5])
    p = O.win_prob_curve(grid, y, comp, N)
    # 무작위 시뮬레이션: 경쟁사 N 명을 F 에서 뽑아 직접 판정
    yy = rng.normal(100, 0.5, 40_000)
    cc = rng.normal(99.9, 0.6, (40_000, N))
    for r, pa in zip(grid, p):
        win = (r >= yy) & ~((cc >= yy[:, None]) & (cc < r)).any(axis=1)
        assert pa == pytest.approx(win.mean(), abs=0.01)


def test_optimal_rate_no_competitors_is_safe_side():
    y = np.random.default_rng(1).normal(100, 0.5, 5000)
    res = O.optimal_rate(y, np.array([100.0]), 0, lo=99, hi=101, step=0.01)
    assert res["win_prob"] == pytest.approx((y <= res["rate"]).mean())
    assert res["valid_prob"] > 0.95                                    # 경쟁이 없으면 하한만 넘기면 된다


def test_more_competitors_lower_win_prob():
    rng = np.random.default_rng(2)
    y, comp = rng.normal(100, 0.5, 5000), rng.normal(100, 0.6, 20000)
    p5 = O.optimal_rate(y, comp, 5)["win_prob"]
    p50 = O.optimal_rate(y, comp, 50)["win_prob"]
    assert p5 > p50 > 0


def test_replay_win():
    assert O.replay_win(100.2, 100.0, np.array([99.9, 100.3]))          # 99.9 는 미달, 100.3 은 나보다 높음
    assert not O.replay_win(100.2, 100.0, np.array([100.1]))            # 100.1 이 [y, r) 에 있음 → 짐
    assert not O.replay_win(99.9, 100.0, np.array([]))                  # 하한 미달


def test_bootstrap_curve_matches_brute_force():
    rng = np.random.default_rng(7)
    auctions = [rng.normal(100.1 + rng.normal(0, 0.3), 0.5, int(rng.integers(3, 15))) for _ in range(30)]
    y = rng.normal(99.9, 0.6, 3000)
    grid = np.array([99.3, 99.9, 100.4])
    p = O.win_prob_curve_bootstrap(grid, y, auctions)
    for r, pb in zip(grid, p):
        brute = np.mean([[O.replay_win(r, yy, c) for yy in y[:400]] for c in auctions])
        assert pb == pytest.approx(brute, abs=0.03)
