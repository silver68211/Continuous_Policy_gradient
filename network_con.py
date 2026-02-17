import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConPolicyGrad(keras.Model):
    """
    Simple continuous policy network for policy-gradient methods.

    Outputs
    -------
    - If learn_var=False: returns mu (mean) only.
    - If learn_var=True : returns (mu, var) where var > 0.

    Notes
    -----
    - Uses two hidden layers with ReLU activations.
    - Variance head uses Softplus to ensure positivity.
    """

    def __init__(
        self,
        fc1_dims: int = 64,
        fc2_dims: int = 64,
        out_mu_dims: int = 1,
        out_var_dims: int = 1,
        learn_var: bool = False,
        min_var: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if fc1_dims < 1 or fc2_dims < 1:
            raise ValueError("fc1_dims and fc2_dims must be >= 1.")
        if out_mu_dims < 1:
            raise ValueError("out_mu_dims must be >= 1.")
        if out_var_dims < 1:
            raise ValueError("out_var_dims must be >= 1.")
        if min_var <= 0:
            raise ValueError("min_var must be > 0.")

        self.learn_var = bool(learn_var)
        self.min_var = float(min_var)

        init = keras.initializers.GlorotUniform()

        self.fc1 = layers.Dense(fc1_dims, activation="relu", kernel_initializer=init)
        self.fc2 = layers.Dense(fc2_dims, activation="relu", kernel_initializer=init)

        # Mean head
        self.mu_head = layers.Dense(out_mu_dims, activation=None, kernel_initializer=init)

        # Variance head (optional)
        self.var_head = None
        if self.learn_var:
            self.var_head = layers.Dense(out_var_dims, activation="softplus", kernel_initializer=init)

    def call(self, state, training=False):
        """
        Parameters
        ----------
        state : tf.Tensor
            Shape (batch, state_dim) or compatible.

        Returns
        -------
        mu : tf.Tensor
            Mean of the policy distribution.
        var : tf.Tensor (optional)
            Variance (positive), returned only if learn_var=True.
        """
        state = tf.convert_to_tensor(state)

        x = self.fc1(state)
        x = self.fc2(x)

        mu = self.mu_head(x)

        if not self.learn_var:
            return mu

        # softplus ensures > 0; add a floor for numerical stability
        var = self.var_head(x) + self.min_var
        return mu, var
