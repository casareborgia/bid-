"""온라인 학습: 밴딧 vs 전정보 (시뮬레이터 3단계).

입찰을 개찰 시각 순서대로 하나씩 겪으면서 "어느 환산 사정률(팔, arm)로 쓸지"를 배운다. 보상 = 1순위면 1, 아니면 0.
  밴딧 피드백  : 내가 쓴 팔의 결과만 안다 (ε-greedy, UCB1, 톰슨 샘플링)
  전정보 피드백: 개찰 후 모든 투찰·예정가격이 공개되므로 모든 팔의 결과를 안다 (Follow-the-Leader, Hedge)
나라장터는 개찰 결과를 공개하므로 실제 상황은 전정보다. 둘을 비교하면 "정보를 얼마나 아는가"가 학습 속도를 얼마나 바꾸는지 보인다.
맥락(context)은 예가 범위(±2/±3)만 쓰고 범위별로 학습기를 따로 둔다(컨텍스트 밴딧은 4단계).

보상 행렬 W[t, k] = 입찰 t 에 팔 k 로 썼다면 1순위였는가(optimize.replay_win, 실제 사정률·실제 경쟁사 투찰).
사후 최적 고정 팔(hindsight best)과의 차이가 후회(regret)다.

사용 예
  uv run python src/sim/bandit.py --months 2026-09,2026-10 --works servc,cnstwk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim import backtest_strategies as B  # noqa: E402
from sim import optimize as O  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = np.round(np.arange(98.5, 101.0 + 1e-9, 0.05), 3)


def reward_matrix(auctions: pd.DataFrame, arms: np.ndarray = ARMS) -> np.ndarray:
    """W[t, k]: 입찰 t(시간순)에 팔 k 로 투찰했다면 1순위였는가."""
    W = np.zeros((len(auctions), len(arms)), dtype=np.int8)
    for t, row in enumerate(auctions.itertuples()):
        c = np.sort(row.rates)
        k = np.searchsorted(c, row.y, side="left")
        nxt = c[k] if k < len(c) else np.inf
        W[t] = (arms >= row.y) & (arms < nxt)
    return W


class Policy:
    full_info = False

    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n, self.rng = n_arms, rng
        self.pulls = np.zeros(n_arms)
        self.wins = np.zeros(n_arms)

    def select(self, t: int) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward_row: np.ndarray) -> None:
        if self.full_info:
            self.pulls += 1
            self.wins += reward_row
        else:
            self.pulls[arm] += 1
            self.wins[arm] += reward_row[arm]

    def _argmax(self, v: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(v == v.max())))


class EpsilonGreedy(Policy):
    def __init__(self, n_arms, rng, eps: float = 0.1):
        super().__init__(n_arms, rng)
        self.eps = eps

    def select(self, t):
        if self.rng.random() < self.eps or self.pulls.sum() == 0:
            return int(self.rng.integers(self.n))
        return self._argmax(self.wins / np.maximum(self.pulls, 1))


class UCB1(Policy):
    def select(self, t):
        untried = np.flatnonzero(self.pulls == 0)
        if len(untried):
            return int(self.rng.choice(untried))
        total = self.pulls.sum()
        return self._argmax(self.wins / self.pulls + np.sqrt(2 * np.log(total) / self.pulls))


class Thompson(Policy):
    def select(self, t):
        return self._argmax(self.rng.beta(1 + self.wins, 1 + self.pulls - self.wins))


class FollowTheLeader(Policy):
    full_info = True

    def select(self, t):
        return self._argmax(self.wins) if self.pulls.sum() else int(self.rng.integers(self.n))


class Hedge(Policy):
    """지수 가중(Exponential Weights). 누적 보상에 비례한 확률로 팔을 고른다."""
    full_info = True

    def __init__(self, n_arms, rng, eta: float = 0.5):
        super().__init__(n_arms, rng)
        self.eta = eta

    def select(self, t):
        w = np.exp(self.eta * (self.wins - self.wins.max()))
        return int(self.rng.choice(self.n, p=w / w.sum()))


POLICIES = {
    "ε-greedy(0.1)": lambda n, r: EpsilonGreedy(n, r, 0.1),
    "UCB1": UCB1,
    "톰슨 샘플링": Thompson,
    "Follow-the-Leader (전정보)": FollowTheLeader,
    "Hedge η=0.5 (전정보)": lambda n, r: Hedge(n, r, 0.5),
}


def run(auctions: pd.DataFrame, policy_factory, seed: int = 0, arms: np.ndarray = ARMS) -> np.ndarray:
    """시간순으로 정책을 돌려 입찰별 보상(0/1)을 돌려준다. 예가 범위별로 학습기를 따로 둔다."""
    W = reward_matrix(auctions, arms)
    rng = np.random.default_rng(seed)
    learners: dict[str, Policy] = {}
    rewards = np.zeros(len(auctions))
    for t, rc in enumerate(auctions["range_cls"]):
        pol = learners.setdefault(rc, policy_factory(len(arms), rng))
        k = pol.select(t)
        rewards[t] = W[t, k]
        pol.update(k, W[t])
    return rewards


def summarize(auctions: pd.DataFrame, seeds: int = 20, arms: np.ndarray = ARMS) -> pd.DataFrame:
    W = reward_matrix(auctions, arms)
    rc = auctions["range_cls"].to_numpy()
    # 사후 최적 고정 팔(예가 범위별) — 미래를 다 알아야 고를 수 있는 기준
    best = np.zeros(len(auctions))
    best_arm = {}
    for g in np.unique(rc):
        m = rc == g
        k = int(W[m].sum(axis=0).argmax())
        best[m], best_arm[g] = W[m, k], float(arms[k])
    label = ", ".join(f"{g} {v:.2f}" for g, v in best_arm.items())
    rows = [{"정책": f"사후 최적 고정 팔 ({label})", "낙찰률": best.mean(), "후회(건)": 0.0}]
    for name, fac in POLICIES.items():
        r = np.array([run(auctions, fac, s, arms) for s in range(seeds)])
        rows.append({"정책": name, "낙찰률": r.mean(), "후회(건)": best.sum() - r.sum(axis=1).mean()})
    rnd = W.mean(axis=1)                                                  # 매번 팔을 무작위로 고를 때
    rows.append({"정책": "팔 무작위", "낙찰률": rnd.mean(), "후회(건)": best.sum() - rnd.sum()})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="밴딧 vs 전정보 온라인 학습")
    ap.add_argument("--months", required=True, help="개찰월 쉼표 나열")
    ap.add_argument("--works", default="servc,cnstwk")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)
    months = sorted(args.months.split(","))
    auctions = B.load_many(months, args.works.split(","))
    tag = months[0] if len(months) == 1 else f"{months[0]}_{months[-1]}"
    df = summarize(auctions, args.seeds)
    df["낙찰률"] = df["낙찰률"].map(lambda v: f"{v*100:.2f}%")
    df["후회(건)"] = df["후회(건)"].map(lambda v: f"{v:.1f}")
    text = (f"# 온라인 학습: 밴딧 vs 전정보 ({', '.join(months)} {args.works} {len(auctions)}건, 시간순, 시드 {args.seeds}개 평균)\n\n"
            f"팔 = 환산 사정률 {ARMS[0]}~{ARMS[-1]}% ({len(ARMS)}개, 0.05 간격). 예가 범위별 학습기.\n\n"
            + df.to_markdown(index=False) + "\n")
    path = ROOT / "out" / f"sim_bandit_{tag}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"리포트: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
