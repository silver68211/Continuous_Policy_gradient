import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

from network_con import ConPolicyGrad

tfd = tfp.distributions


class Agent:
    """
    Simple REINFORCE agent for a 1D continuous action policy.

    - Policy outputs mean (mu) and optionally variance/standard deviation.
    - Actions sampled from Normal(mu, std).
    - Loss: -E[ G_t * log pi(a_t | s_t) ]  (REINFORCE)

    Notes
    -----
    This implementation stores trajectories in memory and updates at episode end.
    """

    def __init__(
        self,
        fc1_dims=256,
        fc2_dims=256,
        out_mu_dims=1,
        out_std_dims=1,
        n_sample=1,
        alpha=1e-3,
        gamma=0.99,
        learn_var=False,
        fixed_std=0.1,
        normalize_returns=True,
        min_std=1e-3,
    ):
        self.gamma = float(gamma)
        self.n_sample = int(n_sample)

        self.learn_var = bool(learn_var)
        self.fixed_std = float(fixed_std)
        self.normalize_returns = bool(normalize_returns)
        self.min_std = float(min_std)

        # trajectory buffers
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        # Policy network
        self.policy = ConPolicyGrad(
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
            out_mu_dims=out_mu_dims,
            out_var_dims=out_std_dims,
            learn_var=self.learn_var,
            min_var=min_std**2,   # if your ConPolicyGrad returns variance
        )
        self.policy.compile(optimizer=Adam(learning_rate=alpha))

    # -------------------------
    # Policy interaction
    # -------------------------
    def _policy_params(self, state):
        """
        Returns (mu, std) tensors.
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        if self.learn_var:
            mu, var = self.policy(state, training=False)
            # ConPolicyGrad earlier returned variance; convert to std
            std = tf.sqrt(tf.maximum(var, self.min_std**2))
        else:
            mu = self.policy(state, training=False)
            std = tf.fill(tf.shape(mu), tf.cast(self.fixed_std, tf.float32))

        return mu, std

    def choose_action(self, state):
        """
        Sample action(s) from the policy. Returns a tensor of shape (n_sample, mu_dim)
        if n_sample>1, otherwise shape (mu_dim,).
        """
        mu, std = self._policy_params(state)

        dist = tfd.Normal(loc=mu, scale=std)

        if self.n_sample == 1:
            action = dist.sample()[0] if len(dist.sample().shape) > len(mu.shape) else dist.sample()
        else:
            # sample n_sample actions; this adds a leading sample dimension
            action = dist.sample(self.n_sample)

        return action

    # -------------------------
    # Reward (example)
    # -------------------------
    def reward(self, action):
        """
        Example shaped reward around a target action.
        Replace with environment reward.
        """
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        mu_target = tf.constant([4.0], dtype=tf.float32)
        target_range = tf.constant([0.5], dtype=tf.float32)
        max_reward = tf.constant([1.0], dtype=tf.float32)

        denom = tf.maximum(target_range, tf.abs(mu_target - action))
        r = max_reward * (target_range / denom)
        return r

    # -------------------------
    # Storage
    # -------------------------
    def store_transition(self, state, action, reward):
        self.state_memory.append(np.array(state, dtype=np.float32))
        self.action_memory.append(np.array(action, dtype=np.float32))
        self.reward_memory.append(float(np.array(reward)))

    # -------------------------
    # Learning (REINFORCE)
    # -------------------------
    def _discounted_returns(self, rewards):
        """
        Compute G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k
        """
        returns = np.zeros(len(rewards), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def learn(self):
        """
        Perform one policy gradient update using stored trajectory.
        Clears memory afterward.
        """
        if len(self.reward_memory) == 0:
            return

        rewards = np.array(self.reward_memory, dtype=np.float32)
        returns = self._discounted_returns(rewards)

        if self.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = tf.convert_to_tensor(np.array(self.state_memory), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.action_memory), dtype=tf.float32)
        returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            mu, std = self._policy_params(states)
            dist = tfd.Normal(loc=mu, scale=std)

            # log_prob per dimension; sum across action dims if needed
            logp = dist.log_prob(actions)
            if len(logp.shape) > 1:
                logp = tf.reduce_sum(logp, axis=-1)

            # REINFORCE loss: -E[ G * log pi(a|s) ]
            loss = -tf.reduce_mean(returns_tf * logp)

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        # clear memory
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

        return float(loss.numpy())
