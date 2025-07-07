from typing import Callable, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.recurrent import RNNCellBase
from recurrent_networks import MaskedRNN

# --- Common Network Utilities ---


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False  # Changed default to False for typical MLP usage
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


# --- Critic Network ---


class QNetwork(nn.Module):

    action_dim: int
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP(
            (*self.hidden_dims, self.action_dim), activations=self.activations
        )(observations)
        return critic


class DoubleQNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states):
        # Creates two Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(
            QNetwork,
            variable_axes={"params": 0},  # Map over parameters for each critic
            split_rngs={
                "params": True
            },  # Use different RNGs for parameter initialization
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs,
        )  # Number of critics to create
        qs = VmapCritic(
            self.action_dim, self.hidden_dims, activations=self.activations
        )(states)
        return qs[0], qs[1]  # Return the two Q-values separately


class RecurrentQNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
    ):
        hidden_state, critic = MaskedRNN(
            self.cell,
            return_carry=True,
        )(
            observations,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        critic = MLP(
            (*self.hidden_dims, self.action_dim), activations=self.activations
        )(critic)
        return hidden_state, critic


class RecurrentDoubleQNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
    ):
        # Creates two Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(
            RecurrentQNetwork,
            variable_axes={"params": 0},  # Map over parameters for each critic
            split_rngs={
                "params": True
            },  # Use different RNGs for parameter initialization
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs,
        )  # Number of critics to create
        hidden_states, qs = VmapCritic(
            self.action_dim,
            self.hidden_dims,
            cell=self.cell,
            activations=self.activations,
        )(states, mask, initial_carry, return_carry_history)
        return hidden_states, qs[0], qs[1]  # Return the two Q-values separately


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):
        # Creates two Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # Map over parameters for each critic
            split_rngs={
                "params": True
            },  # Use different RNGs for parameter initialization
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs,
        )  # Number of critics to create
        qs = VmapCritic(self.hidden_dims, activations=self.activations)(states, actions)
        return qs[0], qs[1]  # Return the two Q-values separately


class RecurrentCritic(nn.Module):
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
    ):
        inputs = jnp.concatenate([observations, actions], -1)
        hidden_state, critic = MaskedRNN(
            self.cell,
            return_carry=True,
        )(
            inputs,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(critic)
        return hidden_state, jnp.squeeze(critic, -1)


class RecurrentVCritic(nn.Module):
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
    ):

        hidden_state, critic = MaskedRNN(
            self.cell,
            return_carry=True,
        )(
            observations,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(critic)
        return hidden_state, jnp.squeeze(critic, -1)


class RecurrentDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
    ):
        # Creates two Critic networks and runs them in parallel.
        VmapCritic = nn.vmap(
            RecurrentCritic,
            variable_axes={"params": 0},  # Map over parameters for each critic
            split_rngs={
                "params": True
            },  # Use different RNGs for parameter initialization
            in_axes=None,  # Inputs (states, actions) are shared
            out_axes=0,  # Stack outputs along the first axis
            axis_size=self.num_qs,
        )  # Number of critics to create
        hidden_states, qs = VmapCritic(
            self.hidden_dims,
            cell=self.cell,
            activations=self.activations,
        )(states, actions, mask, initial_carry, return_carry_history)
        return hidden_states, qs[0], qs[1]  # Return the two Q-values separately


# --- Actor Network ---
class DeterministicActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float  # Needed for TD3's action output clipping/scaling if not handled by scale/bias
    final_fc_init_scale: float
    dropout_rate: float = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training: bool = False):

        outputs = MLP(
            self.hidden_dims,
            activations=nn.relu,
            activate_final=True,
            dropout_rate=self.dropout_rate,
        )(observations, training=training)
        action = (
            nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
            * self.max_action
        )  # Scale output to action range
        return action


class StochasticActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float
    final_fc_init_scale: float
    log_std_min: float
    log_std_max: float
    dropout_rate: float = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ):

        outputs = MLP(
            self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate
        )(observations, training=training)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # TFP uses scale_diag for MultivariateNormalDiag
        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        if self.tanh_squash_distribution:

            return distrax.Transformed(
                base_dist, distrax.Block(distrax.Tanh(), ndims=1)
            )
        else:
            # Returns the raw Normal distribution without tanh squashing.
            return base_dist


class RecurrentStochasticActor(nn.Module):
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    action_dim: int
    max_action: float
    final_fc_init_scale: float
    log_std_min: float
    log_std_max: float
    dropout_rate: float = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
        initial_carry: jnp.ndarray = None,
        return_carry_history: bool = False,
    ):

        outputs = MLP(
            self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate
        )(observations, training=training)

        hidden_state, outputs = MaskedRNN(
            self.cell,
            return_carry=True,
        )(
            outputs,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # TFP uses scale_diag for MultivariateNormalDiag
        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        if self.tanh_squash_distribution:

            return hidden_state, distrax.Transformed(
                base_dist, distrax.Block(distrax.Tanh(), ndims=1)
            )
        else:
            # Returns the raw Normal distribution without tanh squashing.
            return hidden_state, base_dist


class StochasticDiscreteActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float
    dropout_rate: float | None = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ):
        x = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)(
            observations, training=training
        )

        logits = (
            nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(x)
            / temperature
        )

        return distrax.Categorical(logits=logits)


class RecurrentStochasticDiscreteActor(nn.Module):
    hidden_dims: Sequence[int]
    cell: RNNCellBase
    action_dim: int
    final_fc_init_scale: float
    dropout_rate: float | None = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
    ):

        x = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)(
            observations, training=training
        )

        hidden_state, x = MaskedRNN(self.cell, return_carry=True)(
            x,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )

        logits = (
            nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(x)
            / temperature
        )

        return hidden_state, distrax.Categorical(logits=logits)


class Temperature(nn.Module):
    initial_temperature: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        # Parameter for the log of temperature, ensures temperature > 0.
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)
