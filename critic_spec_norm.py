"""
critic_spec_norm.py

Spectral-normalized drop-in replacement for the RLPD critic.

Used only for the causal test in the paper: we want to reduce action-space
sharpness through a mechanism independent of dropout. Spectral normalization
bounds each Dense layer's largest singular value, which bounds the critic's
Lipschitz constant and (through the chain rule) ||grad_a Q||.

This file defines:
  - SpectralDense: Dense layer with spectral-norm reparameterization + learned scale
  - MLPSpecNorm: MLP backbone matching rlpd.networks.MLP signature
  - StateActionValueSpecNorm: single-head critic matching rlpd.networks.StateActionValue

The Ensemble wrapper in rlpd.networks.Ensemble (vmap over heads) composes with
either the upstream StateActionValue or this StateActionValueSpecNorm; no
changes required to the ensemble logic.

USAGE (in sac_learner_v2.py, where the critic is built):

    if getattr(config, 'spec_norm_coef', None) is not None:
        from critic_spec_norm import StateActionValueSpecNorm as CriticCls
        critic_kwargs = {
            'hidden_dims': config.hidden_dims,
            'activations': nn.relu,
            'use_layer_norm': config.critic_layer_norm,
            'dropout_rate': config.critic_dropout_rate,
            'spec_norm_coef': config.spec_norm_coef,
        }
    else:
        from rlpd.networks import StateActionValue as CriticCls
        critic_kwargs = { ... standard kwargs ... }
    critic_base = CriticCls(**critic_kwargs)
    # then Ensemble(critic_base, num=num_qs) as before

The only config change needed is adding `spec_norm_coef` (float or None) to
the agent config. When None, behavior is identical to upstream RLPD.

Spectral-norm implementation uses flax.linen.SpectralNorm, which keeps a
running power-iteration estimate of the largest singular value and divides
the kernel by it at each forward pass. We then multiply by a per-layer
scalar `coef` so the effective Lipschitz constant of each Dense layer is
bounded by `coef` rather than 1.
"""

from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


default_init = nn.initializers.xavier_uniform


class SpectralDense(nn.Module):
    """Dense layer with spectral-normalized kernel, scaled by `coef`.

    Wraps flax.linen.SpectralNorm around a standard Dense. After the
    spectral-norm reparameterization, the layer's Lipschitz constant is
    bounded by `coef`. The bias is unaffected (biases don't contribute to
    the gradient norm w.r.t. inputs).

    For the causal test, we set coef identically across all layers in the
    MLP backbone. Three coefficients are swept: {0.1, 1.0, 10.0}. These
    bound the total critic Lipschitz constant (in the jointly-measured
    state + action space) by coef^(num_layers).
    """

    features: int
    coef: float = 1.0
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # SpectralNorm wraps the inner Dense. During train=True, the
        # singular-value estimate is updated via one power-iteration step
        # and the kernel is divided by it. During train=False, the current
        # estimate is used without updating.
        dense = nn.Dense(features=self.features, kernel_init=self.kernel_init)
        dense = nn.SpectralNorm(dense)
        y = dense(x, update_stats=train)
        return self.coef * y


class MLPSpecNorm(nn.Module):
    """Spectral-normalized MLP matching the signature of rlpd.networks.MLP.

    Each hidden Dense is replaced with a SpectralDense. LayerNorm and
    dropout positions match the upstream MLP exactly, so the only
    behavioural difference relative to upstream is the spectral
    reparameterization of the Dense kernels.
    """

    hidden_dims: Sequence[int]
    activations: Callable = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    spec_norm_coef: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = SpectralDense(
                features=size,
                coef=self.spec_norm_coef,
            )(x, train=training)

            # Apply activation + layernorm + dropout on all hidden layers, and on
            # the final layer only if activate_final=True. This matches the
            # upstream MLP convention.
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0.0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class StateActionValueSpecNorm(nn.Module):
    """Spectral-normalized critic head.

    Drop-in replacement for rlpd.networks.StateActionValue. The signature
    (observations, actions, training) and output shape match upstream.

    The output Dense that projects to a scalar Q value is also
    spectrally-normalized, to avoid leaving a large-Lipschitz layer at the
    top that would defeat the purpose of the regularization below it.
    """

    hidden_dims: Sequence[int]
    activations: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    spec_norm_coef: float = 1.0

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)

        h = MLPSpecNorm(
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            spec_norm_coef=self.spec_norm_coef,
        )(inputs, training=training)

        # Scalar head, also spectrally-normalized. A non-spec-normed top
        # layer could set the critic's effective Lipschitz arbitrarily high
        # regardless of what the backbone does.
        q = SpectralDense(features=1, coef=self.spec_norm_coef)(h, train=training)
        return jnp.squeeze(q, -1)
