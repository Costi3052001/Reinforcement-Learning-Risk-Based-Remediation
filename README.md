# EPSS × CVSS Vulnerability Triage Simulation

This project simulates a simple vulnerability triage workflow using a custom Gymnasium environment. It compares two decision strategies across multiple episodes:

- **Monte Carlo (random):** selects a random vulnerability from the backlog
- **Rule-based (EPSS × CVSS product):** selects the vulnerability with the highest `EPSS × CVSS` score

The goal is to evaluate whether a simple heuristic policy performs better than random selection when prioritizing vulnerabilities.

## Project Overview

Each vulnerability is represented by:

- **EPSS**: estimated likelihood of exploitation
- **CVSS base score**: severity score, normalized to a 0–1 range

At each step:

1. A backlog of vulnerabilities is shown
2. The agent chooses one vulnerability
3. A reward is assigned as:

```python
reward = EPSS * CVSS
```

4. The chosen vulnerability is replaced with a newly sampled one
5. The process repeats for a fixed number of steps

The episode return is the total accumulated reward across all steps in the episode.

## Features

- Loads vulnerability data from a CSV catalog
- Builds a custom **Gymnasium** environment for triage simulation
- Runs repeated episodes with controlled random seeds
- Compares:
  - random policy
  - rule-based priority policy
- Saves result visualizations:
  - line chart of episode returns
  - normal distribution curves of returns

## Project Structure

```text
project/
│
├── epss_cvss.py
├── data/
│   ├── catalog_2023_2025_cvss.csv
│   ├── results_line.png
│   └── results_normal_curves.png
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scipy gymnasium
```

## Dataset Requirements

The script expects the following CSV file:

```text
data/catalog_2023_2025_cvss.csv
```

It must include at least these columns:

- `epss`
- `cvss_base`

Example:

```csv
epss,cvss_base
0.42,8.8
0.10,5.6
0.75,9.1
```

## How It Works

### 1. Load the catalog
The script reads EPSS and CVSS values from the CSV file and converts them into a NumPy array.

### 2. Create the environment
A custom `VulnTriageEnv` environment is used with:

- **observation space**: flattened backlog of vulnerability pairs
- **action space**: index of the selected vulnerability in the backlog

### 3. Run policies

#### Random policy
Chooses any backlog item uniformly at random.

#### Rule-based policy
Computes:

```python
score = epss * cvss
```

and selects the item with the highest score.

### 4. Compare returns
For each episode, the total reward is recorded and plotted.

## Default Experiment Settings

The script currently uses:

```python
backlog_size = 12
steps = 10
episodes = 100
RNG_SEED = 42
```

## Running the Project

Run:

```bash
python epss_cvss.py
```

## Output

When executed, the script:

1. Prints summary statistics for both strategies
2. Saves a line chart to:

```text
data/results_line.png
```

3. Saves normal distribution comparison curves to:

```text
data/results_normal_curves.png
```

## Example Result Interpretation

- If the **rule-based EPSS × CVSS** policy shows a higher mean return than the random policy, it suggests that simple heuristic prioritization is more effective than random selection.
- The line plot shows how episode returns vary across runs.
- The normal curve plot gives a visual comparison of the distributions of returns for both strategies.

## Main Components

### `load_catalog(catalog_path)`
Loads and formats the vulnerability dataset.

### `VulnTriageEnv`
Custom Gym environment for backlog-based triage.

### `random_policy(obs, action_space)`
Baseline random selection strategy.

### `rule_policy_product(backlog_size)`
Heuristic strategy that selects the max `EPSS × CVSS` item.

### `run(policy, catalog, backlog_size, steps, episodes, seed0)`
Executes repeated episodes and returns total episode rewards.

### `main()`
Runs the full experiment, prints statistics, and saves plots.

## Notes

- CVSS scores are normalized by dividing `cvss_base` by `10.0`
- Rewards are based only on the product of EPSS and CVSS
- The environment samples vulnerabilities randomly from the catalog
- The current implementation is a simulation baseline and can be extended with reinforcement learning agents later

## Possible Future Improvements

- Add a learned RL policy for comparison
- Include additional vulnerability features
- Add statistical significance tests to compare strategies
- Track cumulative averages across episodes
- Add bar charts or box plots for easier comparison

## License

Add your preferred license here, for example:

```text
MIT License
```
