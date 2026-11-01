"""입찰 환경 시뮬레이터(sim.env) 테스트. 규칙 구조와 이론값을 확인한다(실데이터 불필요)."""
import numpy as np
import pytest

from sim import env as E


@pytest.mark.parametrize("rc", [2, 3])
def test_reserve_deviations_layout(rc):
    rng = np.random.default_rng(0)
    dev = E.reserve_deviations(rc, rng, size=2000)
    assert dev.shape == (2000, 15)
    assert (dev >= -rc).all() and (dev <= rc).all()
    col = 0
    for lo, hi, k in E.LAYOUT[rc]:                                     # 칸마다 정확히 1개 (열은 구간 순서대로)
        edges = np.linspace(lo, hi, k + 1)
        part = dev[:, col:col + k]
        assert (np.floor((part - lo) / (edges[1] - edges[0])) == np.arange(k)).all()
        col += k
    if rc == 3:
        assert ((dev < 0).sum(axis=1) == 8).all()                      # 실측: 아래 8개 / 위 7개


def test_reserve_deviations_rejects_other_range():
    with pytest.raises(ValueError):
        E.reserve_deviations(4, np.random.default_rng(0))


def test_draw_numbers_four_distinct_and_uniform():
    rng = np.random.default_rng(1)
    idx = E.draw_numbers(25, rng, size=3000)
    assert idx.shape == (3000, 4)
    assert all(len(set(r)) == 4 for r in idx)
    share = np.bincount(idx.ravel(), minlength=15) / idx.size * 15
    assert np.allclose(share, 1.0, atol=0.08)                          # 번호 선택은 균등


def test_simulated_rates_match_theory():
    """β=0 이면 ±2% 평균 100, 표준편차 이론값 √(σ²/4·11/14) ≈ 0.51 (σ² ≈ 16/12)."""
    rng = np.random.default_rng(2)
    r2 = E.simulate_yega_rates(2, 200_000, rng, draw_bias=0.0)
    assert r2.mean() == pytest.approx(100.0, abs=0.01)
    assert r2.std() == pytest.approx(np.sqrt(16 / 12 / 4 * 11 / 14), abs=0.01)
    r3 = E.simulate_yega_rates(3, 200_000, rng, draw_bias=0.0)
    assert r3.mean() == pytest.approx(100 + (8 * -1.5 + 7 * 1.5) / 15, abs=0.01)  # 아래 8 / 위 7 → 평균 99.9


def test_draw_bias_lowers_mean():
    rng = np.random.default_rng(3)
    base = E.simulate_yega_rates(3, 200_000, rng, draw_bias=0.0).mean()
    biased = E.simulate_yega_rates(3, 200_000, rng).mean()             # 기본 β=0.05
    assert base - biased == pytest.approx(0.046, abs=0.012)


def test_agent_draws_close_to_direct_sampling():
    rng = np.random.default_rng(4)
    a = E.simulate_yega_rates(2, 8000, rng, n_bidders=20)
    b = E.simulate_yega_rates(2, 8000, rng, draw_bias=0.0)
    assert abs(a.mean() - b.mean()) < 0.03 and abs(a.std() - b.std()) < 0.03


def test_run_auction_winner_is_lowest_valid_bid():
    rng = np.random.default_rng(5)
    base = 100_000_000
    bids = np.array([80_000_000, 88_000_000, 89_000_000, 95_000_000])
    res = E.run_auction(base, 2, bids, 87.745, rng)
    assert res.lower_limit == pytest.approx(res.yega * 0.87745)
    assert list(res.valid) == list(bids >= res.lower_limit)
    assert res.winner == int(np.flatnonzero(res.valid)[0])             # 정렬돼 있으므로 첫 유효 투찰
    assert E.run_auction(base, 2, np.array([1.0]), 87.745, rng).winner is None


def test_lower_limit_amount_with_a_value():
    assert E.lower_limit_amount(550_000_000, 89.745, 30_000_000) == pytest.approx(496_674_000)


def test_empirical_rates_sampler():
    emp = E.EmpiricalRates({2: [99.0, 101.0], 3: [98.0]})
    rng = np.random.default_rng(6)
    assert set(emp.sample(2, 100, rng)) == {99.0, 101.0}
    assert (emp.sample(3, 5, rng) == 98.0).all()
