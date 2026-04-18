from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, norm, t as t_dist
import gymnasium as gym
from gymnasium import spaces

RNG_SEED = 42

def load_catalog(catalog_path: Path):
    df = pd.read_csv(catalog_path)
    epss_list = df["epss"].astype(float).tolist()
    cvss_list = (df["cvss_base"].astype(float) / 10.0).tolist()
    data = []
    for i in range(len(epss_list)):
        e = epss_list[i]
        c = cvss_list[i]
        data.append([e, c])
    return np.array(data, dtype=float)
class VulnTriageEnv(gym.Env):
    def __init__(self, catalog, backlog_size: int, steps: int, seed: int): #calling gym constructir
        super().__init__()
        self.catalog = catalog
        self.backlog_size = int(backlog_size)
        self.steps = int(steps)
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.backlog_size * 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.backlog_size)
        self._t = 0
        self.backlog = None
    def _sample_row(self):
        idx = self._rng.integers(0, len(self.catalog))
        return self.catalog[idx]
    def _obs(self):
        return self.backlog.astype(np.float32).ravel() #made an error here, i must return 1D array
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        rows = []
        for _ in range(self.backlog_size):
            rows.append(self._sample_row())
        self.backlog = np.array(rows, dtype=float)
        return self._obs()
    def step(self, action: int):
        e, c = self.backlog[action]
        reward = float(e * c)
        self.backlog[action] = self._sample_row()
        self._t += 1 #not to exceed # of steps
        terminated = self._t >= self.steps
        return self._obs(), reward, terminated, False, {}
def random_policy(obs, action_space):
    return action_space.sample() #returns random sample

def rule_policy_product(backlog_size: int):
    def policy(obs, action_space):
        pairs = np.asarray(obs, dtype=float).reshape(backlog_size, 2)
        e = pairs[:, 0]   #that's our epss (liklehood)
        c = pairs[:, 1]   #impact
        scores = e * c
        return int(np.argmax(scores)) #returns the highest score
    return policy

def run(policy, catalog, backlog_size, steps, episodes, seed0=RNG_SEED):
    rets = []
    for ep in range(episodes):
        env = VulnTriageEnv(catalog, backlog_size=backlog_size, steps=steps, seed=seed0 + ep)
        obs = env.reset() #starting our env
        G = 0.0 #adds rewards
        done = False
        while not done: #when all steps are done, becomes true
            a = policy(obs, env.action_space)
            obs, r, done, _, _ = env.step(a)
            G += r
        rets.append(G)
    return np.array(rets, dtype=float)

def main():
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"
    catalog_path = DATA / "catalog_2023_2025_cvss.csv"
    backlog_size = 12
    steps = 10
    episodes = 100
    MC = "Monte Carlo (random)"
    RB = "Rule-based (EPSS×CVSS product)"
    catalog = load_catalog(catalog_path)
    rand_scores = run(random_policy, catalog, backlog_size, steps, episodes, seed0=RNG_SEED)
    rule_scores = run(rule_policy_product(backlog_size), catalog, backlog_size, steps, episodes, seed0=RNG_SEED)
    nA = len(rand_scores)
    meanA = np.mean(rand_scores)
    sdA = np.std(rand_scores, ddof=1)
    nB = len(rule_scores)
    meanB = np.mean(rule_scores)
    sdB = np.std(rule_scores, ddof=1)

    print("\n results:\n")
    print(f"{MC:>28}: n={nA}, mean={meanA:.4f}, sd={sdA:.4f})")
    print(f"{RB:>28}: n={nB}, mean={meanB:.4f}, sd={sdB:.4f})")
    plt.figure()
    plt.plot(rand_scores, label=MC, alpha=0.7)
    plt.plot(rule_scores, label=RB, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("Episode Returns Across Runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA / "results_line.png", dpi=140)
    plt.clf()
    fig, ax = plt.subplots()
    x = np.linspace(min(rand_scores.min(), rule_scores.min()),
                    max(rand_scores.max(), rule_scores.max()), 200)
    ax.plot(x, norm.pdf(x, meanA, sdA), color="blue", label=f"{MC} (mean={meanA:.2f}, sd={sdA:.2f})")
    ax.axvline(meanA, color="blue", linestyle="dashed")
    ax.plot(x, norm.pdf(x, meanB, sdB), color="green", label=f"{RB} (mean={meanB:.2f}, sd={sdB:.2f})")
    ax.axvline(meanB, color="green", linestyle="dashed")
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA / "results_normal_curves.png", dpi=140)
    plt.close(fig)
if __name__ == "__main__":
    main()
