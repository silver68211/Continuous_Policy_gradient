# Continuous REINFORCE (TensorFlow) — Mean/Variance Policy Network

A lightweight implementation of **REINFORCE (policy gradient)** for a **continuous action space** using TensorFlow / Keras.
The policy network outputs an action **mean** and optionally a learned **variance** (or std), and the agent updates parameters using the Monte Carlo return.

This repo is intended as a **minimal research/educational baseline** you can adapt to your own environments.

---

## Features

* ✅ Continuous-action REINFORCE agent
* ✅ Policy network with configurable MLP backbone
* ✅ Optional **learned variance** (stochastic policy with trainable uncertainty)
* ✅ Uses **TensorFlow Probability** (`tfd.Normal`) for stable log-prob computation
* ✅ Simple training script + learning curve plotting

---

## Repository Structure (Suggested)

```
.
├── network_con.py        # ConPolicyGrad policy network (mu + optional var/std)
├── con_reinforce.py      # Agent (REINFORCE) with trajectory memory + update
├── train.py              # Training loop (example)
└── utils.py              # plot_learning() helper
```

If your filenames differ, update the import paths in `train.py`.

---

## Installation

### Requirements

* Python 3.9+
* TensorFlow 2.x
* TensorFlow Probability
* NumPy
* Matplotlib (for plots)

Install:

```bash
pip install tensorflow tensorflow-probability numpy matplotlib
```

---

## Quick Start

Run training:

```bash
python con_main.py
```

This will generate plots:

* `score.png` — running-average episode reward
* `mu.png` — running-average mean estimate
* `sigma.png` — running-average std/variance behavior (if enabled)

---

## How It Works

### Policy Network (`ConPolicyGrad`)

The policy is a simple MLP:

* Two hidden layers (ReLU)
* Output head for **mean** `mu`
* Optional output head for **variance/std** (positive via Softplus)

When `learn_var=True`, the network returns:

```python
mu, var
```

Otherwise it returns:

```python
mu
```

---

### Agent (`Agent`)

The agent stores an episode trajectory:

* states
* actions
* rewards

Then computes Monte Carlo returns:

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t} r_k
$$

And applies REINFORCE:

$$
\mathcal{L}(\theta) = -\mathbb{E}\left[G_t \log \pi_\theta(a_t|s_t)\right]
$$

The log-probability is computed using:

```python
tfd.Normal(loc=mu, scale=std).log_prob(action)
```

---

## Configuration

You can customize training via:

* `learn_var`: learn policy variance (True/False)
* `fixed_std`: use constant std when `learn_var=False`
* `gamma`: discount factor
* `alpha`: learning rate
* layer sizes (`fc1_dims`, `fc2_dims`, etc.)

Example:

```python
agent = Agent(
    alpha=3e-3,
    gamma=0.99,
    learn_var=True,
    fixed_std=0.1,
)
```

---

## Notes / Common Pitfalls

* **Returns computation**: Make sure returns are computed from time `t` onward (not always from 0).
* **Shape consistency**: Use consistent shapes for states/actions, ideally `(batch, dim)` for network inputs.
* **Variance stability**: Enforce a minimum variance/std to avoid numerical issues (e.g., `min_std=1e-3`).

---

## Extending to Real Environments

This repo currently uses a toy reward function.
To integrate with Gym / custom environment:

1. Replace `reward()` with `env.step(action)`
2. Store transitions per step
3. Call `learn()` after each episode

Pseudo-code:

```python
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.store_transition(state, action, reward)
    state = next_state
agent.learn()
```

---

## License

MIT License (recommended for reuse).

---

## Acknowledgments

This project is inspired by classic REINFORCE / policy gradient baselines and uses TensorFlow Probability distributions for stable log-prob evaluation.

---

