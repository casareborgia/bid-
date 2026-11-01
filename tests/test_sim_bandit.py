"""온라인 학습(sim.bandit) 테스트: 보상 행렬, 밴딧·전정보 정책이 좋은 팔을 찾는지."""
import numpy as np
import pandas as pd

from sim import bandit as BD
from sim import optimize as O


def _auctions(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = rng.normal(99.9, 0.6)
        rows.append({"bid_key": f"B{i}", "range_cls": "±3", "y": y, "n": 8,
                     "open_dt": pd.Timestamp("2026-10-01") + pd.Timedelta(hours=i), "rates": rng.normal(100.3, 0.5, 8)})
    return pd.DataFrame(rows)


def test_reward_matrix_matches_replay():
    a = _auctions(40)
    W = BD.reward_matrix(a)
    for t in range(0, 40, 7):
        row = a.iloc[t]
        assert [bool(x) for x in W[t]] == [O.replay_win(r, row.y, row.rates) for r in BD.ARMS]


def test_full_info_learns_at_least_as_well_as_bandit():
    a = _auctions()
    W = BD.reward_matrix(a)
    best = W.sum(axis=0).max() / len(a)
    ftl = BD.run(a, BD.FollowTheLeader, seed=1).mean()
    rnd = W.mean()
    assert ftl > rnd and ftl > best - 0.05                            # 전정보는 금방 최적 근처
    ts = np.mean([BD.run(a, BD.Thompson, seed=s).mean() for s in range(3)])
    assert ts > rnd                                                   # 밴딧도 무작위보다는 낫다


def test_summarize_has_all_policies():
    df = BD.summarize(_auctions(150), seeds=2)
    assert len(df) == len(BD.POLICIES) + 2 and df["후회(건)"].iloc[0] == 0
