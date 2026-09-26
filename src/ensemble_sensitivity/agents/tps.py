# Copied from / derived from ikostrikov/rlpd (MIT, Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith); see LICENSE-rlpd.
"""SACLearnerV2 + TD3-style target-policy smoothing (TPS).

Next actions are perturbed with clipped Gaussian noise before target-critic
evaluation (Fujimoto et al. 2018).

a'_used = clip(a' + clip(eps, -c, c), -1, 1),  eps ~ N(0, sigma^2),
c = 2.5 * sigma (TD3's 0.2/0.5 ratio), actions rescaled to [-1, 1] by
wrap_gym. sigma = 0 removes the perturbation, but this implementation still
consumes additional random keys compared with SACLearnerV2.

If backup_entropy were True, log-probs are computed at the UNPERTURBED
sampled action (perturbed actions can sit on the tanh boundary where
log_prob diverges); our runs use backup_entropy=False.
"""
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct
from rlpd.networks import subsample_ensemble

from ensemble_sensitivity.agents.sac import SACLearnerV2


class SACLearnerV2TPS(SACLearnerV2):
    target_smoothing_sigma: float = struct.field(
        pytree_node=False, default=0.0)

    @classmethod
    def create(cls, seed, observation_space, action_space,
               target_smoothing_sigma=0.0, **kwargs):
        if float(target_smoothing_sigma) > 0 and kwargs.get(
                "backup_entropy", True):
            raise ValueError(
                "TPS + backup_entropy=True is undefined (entropy term at "
                "the unperturbed action vs Q at the perturbed one); pass "
                "backup_entropy=False.")
        agent = super().create(
            seed, observation_space, action_space, **kwargs)
        agent = agent.replace(
            target_smoothing_sigma=float(target_smoothing_sigma))
        assert type(agent) is cls and (
            agent.target_smoothing_sigma == float(target_smoothing_sigma)), \
            "replace() lost the TPS subclass type or the sigma field"
        return agent

    def update_critic(self, batch):
        # Identical to SACLearnerV2.update_critic except the marked block.
        dist = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"])
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)
        next_actions_raw = next_actions

        # --- TPS: clipped-noise smoothing of the target evaluation point ---
        if self.target_smoothing_sigma > 0:
            sigma = self.target_smoothing_sigma
            noise_clip = 2.5 * sigma
            key_n, rng = jax.random.split(rng)
            noise = jnp.clip(
                jax.random.normal(key_n, next_actions.shape) * sigma,
                -noise_clip, noise_clip)
            next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        # -------------------------------------------------------------------

        key, rng = jax.random.split(rng)

        if self.independent_targets:
            next_qs = self.critic.apply_fn(
                {"params": self.target_critic.params},
                batch["next_observations"],
                next_actions, True,
                rngs={"dropout": key})
            target_q = (batch["rewards"]
                        + self.discount * batch["masks"] * next_qs)
        else:
            target_params = subsample_ensemble(
                key, self.target_critic.params,
                self.num_min_qs, self.num_qs)
            key, rng = jax.random.split(rng)
            next_qs = self.target_critic.apply_fn(
                {"params": target_params},
                batch["next_observations"],
                next_actions, True,
                rngs={"dropout": key})
            next_q = next_qs.min(axis=0)
            target_q = (batch["rewards"]
                        + self.discount * batch["masks"] * next_q)

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions_raw)
            target_q = target_q - (
                self.discount * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs)

        key, rng = jax.random.split(rng)
        mask_key, rng = jax.random.split(rng)
        do_bootstrap = self.bootstrap_mask

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"], True,
                rngs={"dropout": key})
            td_errors = (qs - target_q) ** 2

            if do_bootstrap:
                bmask = jax.random.bernoulli(
                    mask_key, p=0.5, shape=td_errors.shape)
                critic_loss = ((bmask * td_errors).sum()
                               / jnp.maximum(bmask.sum(), 1.0))
            else:
                critic_loss = td_errors.mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean()}

        grads, info = jax.grad(
            critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(
            params=target_critic_params)
        return self.replace(
            critic=critic, target_critic=target_critic, rng=rng), info
