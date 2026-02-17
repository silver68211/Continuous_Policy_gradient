import numpy as np
import tensorflow as tf

from con_reinforce import Agent
from utils import plot_learning  # use the improved version name if you renamed it


def main():
    # -----------------------------
    # Config
    # -----------------------------
    learn_var = True          # if True: policy outputs (mu, var/std); else: only mu
    fixed_std = 0.1           # used only when learn_var=False (or as fallback)
    num_episodes = 10_000
    window = 100

    # -----------------------------
    # Agent
    # -----------------------------
    agent = Agent(
        alpha=0.003,
        gamma=1.0,
        learn_var=learn_var,
        fixed_std=fixed_std,  # adjust to your Agent signature
    )

    # -----------------------------
    # Logging
    # -----------------------------
    score_history = []
    mu_history = []
    sigma_history = []

    # Dummy constant initial state (toy example)
    init_state = tf.constant([[1.0]], dtype=tf.float32)

    for ep in range(num_episodes + 1):
        state = init_state
        score = 0.0

        # One-step episode (your original code effectively does one step)
        if learn_var:
            mu, std = agent.step(state)
        else:
            mu = agent.step(state)
            std = fixed_std

        action = agent.choose_action(mu=mu, std=std)
        reward = agent.reward(action=action)

        agent.store_transitions(state, action, reward)
        agent.learn()

        # Logging
        score += float(np.array(reward))
        score_history.append(score)

        # store scalars for plotting (convert tensors cleanly)
        mu_history.append(float(tf.reshape(mu, [-1])[0].numpy()))
        sigma_history.append(float(tf.reshape(std, [-1])[0].numpy()) if tf.is_tensor(std) else float(std))

        avg_score = np.mean(score_history[-window:])
        print(f"episode: {ep:5d} | score: {score:8.4f} | avg({window}): {avg_score:8.4f}")

    # -----------------------------
    # Plots
    # -----------------------------
    plot_learning(score_history, filename="score.png", window=window, ylabel="Score")
    plot_learning(mu_history, filename="mu.png", window=window, ylabel="Mean")
    plot_learning(sigma_history, filename="sigma.png", window=window, ylabel="Sigma")


if __name__ == "__main__":
    main()
