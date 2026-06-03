import csv
import os
import pickle
import time
from collections import deque
from datetime import datetime
from functools import partial

import flax
import hydra
import jax
import jax.numpy as jp
import jax.random as rax
import numpy as np
import optax
from jax import config
from omegaconf import DictConfig, OmegaConf

import wandb
from coin_agent import CoinAgent
from coin_game import CoinGame, coin_game_params, episode_stats, make_zero_episode
from coin_train import (
    Optimizer,
    TRAIN_ITERATION_METRICS_FIELDNAMES,
    compute_iteration_metric_row,
    format_wandb_metrics,
    get_metrics_csv_filename,
    get_metrics_log_timestep_freq,
    get_num_training_episodes_per_iteration,
    get_num_training_iterations,
    get_num_training_timesteps_per_iteration,
    get_total_training_episodes,
    get_total_training_timesteps,
    initialize_wandb,
    play_episode_scan_inner_gru,
    scalar_mean,
    tree_concatenate,
)
from reciprocator_coin_agent import (
    CoinStateValuePredictor,
    ConvGRUActorCriticCoinAgent,
    DiscreteTransitionPredictor,
    GRUActorCriticCoinAgent,
    ScalarPredictor,
)
from utils import clip_grads_by_norm, global_norm, npify, rscope


RECIPROCAL_REWARD_TYPES = {
    'petty',
    'signed_grudge',
    'petty_payoff',
    'signed_petty_payoff',
    'signed_grudge_petty_payoff',
    'grudge_minus_debit',
    'signed_grudge_minus_debit',
}

RECIPROCAL_METRIC_FIELDNAMES = [
    'loss_ppo_total',
    'loss_ppo_policy',
    'loss_ppo_value',
    'ppo_entropy',
    'grad_ppo_norm',
    'policy_anchor_kl',
    'policy_anchor_loss',
    'policy_anchor_active',
    'policy_anchor_score',
    'policy_anchor_best_score',
    'policy_anchor_update',
    'loss_voi_state_value',
    'loss_voi_reward',
    'loss_voi_transition',
    'loss_voi_cf_reward0',
    'loss_voi_cf_reward1',
    'loss_voi_cf_transition0',
    'loss_voi_cf_transition1',
    'mean_reciprocal_reward_player0',
    'mean_reciprocal_reward_player1',
    'std_reciprocal_reward_player0',
    'std_reciprocal_reward_player1',
    'mean_abs_reciprocal_reward_player0',
    'mean_abs_reciprocal_reward_player1',
    'mean_cumulative_reciprocal_reward_player0',
    'mean_cumulative_reciprocal_reward_player1',
    'avg_env_return_player0',
    'avg_env_return_player1',
    'avg_shaped_return_player0',
    'avg_shaped_return_player1',
    'corr_reciprocal_reward_same_color_pickup_player0',
    'corr_reciprocal_reward_same_color_pickup_player1',
    'corr_reciprocal_reward_other_color_pickup_player0',
    'corr_reciprocal_reward_other_color_pickup_player1',
    'mean_end_grudge_player0',
    'mean_end_grudge_player1',
    'mean_voi_0_on_1',
    'mean_voi_1_on_0',
    'action_frequency_left_player0',
    'action_frequency_right_player0',
    'action_frequency_up_player0',
    'action_frequency_down_player0',
    'action_frequency_left_player1',
    'action_frequency_right_player1',
    'action_frequency_up_player1',
    'action_frequency_down_player1',
]
RECIPROCAL_CSV_FIELDNAMES = [
    *TRAIN_ITERATION_METRICS_FIELDNAMES,
    *[field for field in RECIPROCAL_METRIC_FIELDNAMES if field not in TRAIN_ITERATION_METRICS_FIELDNAMES],
]


def categorical_entropy(logp):
    return -jp.sum(jp.nan_to_num(jp.exp(logp) * logp), axis=-1)


def safe_corr(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.std() * y.std() + 1e-8)


def discounted_returns_2d(rewards, gamma):
    rewards_time_major = rewards.T

    def body(carry, reward_t):
        value = reward_t + gamma * carry
        return value, value

    _, returns_reversed = jax.lax.scan(
        body,
        jp.zeros_like(rewards_time_major[0]),
        jp.flip(rewards_time_major, axis=0),
    )
    return jp.flip(returns_reversed, axis=0).T


def discounted_returns_3d(rewards, gamma):
    rewards_time_major = jp.swapaxes(rewards, 0, 1)

    def body(carry, reward_t):
        value = reward_t + gamma * carry
        return value, value

    _, returns_reversed = jax.lax.scan(
        body,
        jp.zeros_like(rewards_time_major[0]),
        jp.flip(rewards_time_major, axis=0),
    )
    return jp.swapaxes(jp.flip(returns_reversed, axis=0), 0, 1)


@jax.jit
def build_influence_training_data(episodes, gamma):
    states = episodes['obs'][:, :-1, 0]
    joint_actions = episodes['act']
    joint_rewards = episodes['rew']
    joint_returns = discounted_returns_3d(joint_rewards, gamma)

    batch_size, trace_length = joint_actions.shape[:2]
    states_flat = states.reshape(batch_size, trace_length, -1)
    time_remaining = (
        (trace_length - 1 - jp.arange(trace_length, dtype=states.dtype))
        .reshape(1, trace_length, 1)
        .repeat(batch_size, axis=0)
    )
    full_inputs = jp.concatenate(
        [joint_actions.astype(states.dtype), states_flat, time_remaining],
        axis=-1,
    )
    cf_inputs0 = jp.concatenate(
        [joint_actions[:, :, 1:2].astype(states.dtype), states_flat, time_remaining],
        axis=-1,
    )
    cf_inputs1 = jp.concatenate(
        [joint_actions[:, :, 0:1].astype(states.dtype), states_flat, time_remaining],
        axis=-1,
    )
    width = states.shape[-1]
    num_position_classes = states.shape[-2] * width
    next_player0_position = episodes['player1_pos'][:, 1:-1]
    next_player1_position = episodes['player2_pos'][:, 1:-1]
    next_coin_position = episodes['coin_pos'][:, 1:-1]
    next_coin_owner = episodes['coin_owner'][:, 1:-1, 0]

    def flatten_position(position):
        return position[..., 0] * width + position[..., 1]

    transition_labels = jp.stack(
        [
            flatten_position(next_player0_position),
            flatten_position(next_player1_position),
            next_coin_owner * num_position_classes + flatten_position(next_coin_position),
        ],
        axis=-1,
    )

    return {
        'state_inputs': states.reshape(batch_size * trace_length, -1),
        'full_inputs': full_inputs.reshape(batch_size * trace_length, -1),
        'cf_inputs0': cf_inputs0.reshape(batch_size * trace_length, -1),
        'cf_inputs1': cf_inputs1.reshape(batch_size * trace_length, -1),
        'transition_full_inputs': full_inputs[:, :-1].reshape(batch_size * (trace_length - 1), -1),
        'transition_cf_inputs0': cf_inputs0[:, :-1].reshape(batch_size * (trace_length - 1), -1),
        'transition_cf_inputs1': cf_inputs1[:, :-1].reshape(batch_size * (trace_length - 1), -1),
        'state_value_labels': joint_returns.reshape(batch_size * trace_length, -1),
        'reward_labels': joint_rewards.reshape(batch_size * trace_length, -1),
        'transition_labels': transition_labels.reshape(batch_size * (trace_length - 1), -1),
    }


@partial(jax.jit, static_argnames=('player',))
def build_ppo_training_data(episodes, rewards, gamma, player):
    batch_size, trace_length = episodes['act'].shape[:2]
    obs_seq = episodes['obs'][:, :-1, player].reshape(batch_size, trace_length, -1)
    action_seq = episodes['act'][:, :, player]

    taken_logps = jp.take_along_axis(episodes['logp'], episodes['act'][..., None], axis=-1)[..., 0]
    old_logp_seq = taken_logps[:, :, player]
    reward_seq = rewards[:, :, player]
    return_seq = discounted_returns_2d(reward_seq, gamma)
    return_seq = (return_seq - return_seq.mean()) / (return_seq.std() + 1e-5)

    return {
        'obs_seq': obs_seq,
        'action_seq': action_seq,
        'old_logp_seq': old_logp_seq,
        'return_seq': return_seq,
    }


@partial(jax.jit, static_argnames=('hp',))
def generate_selfplay_episodes(agent0, agent1, carries, rng, hp):
    rngs = rax.split(rscope(rng, 'reciprocator_gen_episode'), hp['batch_size'])
    init_envs, _ = jax.vmap(lambda r: CoinGame.init(
        rng=r,
        **coin_game_params(hp),
    ))(rscope(rngs, 'game_init'))

    def play_one_episode(play_rng, env):
        return play_episode_scan_inner_gru(
            dict(
                agent0=agent0,
                agent1=agent1,
                rng=play_rng,
                t=0,
                **carries,
            ),
            trace_length=hp['game']['game_length'],
            env=env,
        )

    episodes, aux = jax.vmap(play_one_episode)(rscope(rngs, 'play_rng'), init_envs)
    return episodes


def make_predictor_train_fn(model, optimizer, batch_size, num_steps):
    @jax.jit
    def train(params, opt_state, rng, inputs, labels):
        num_samples = inputs.shape[0]

        def loss_fn(p, x, y):
            pred = model.apply(p, x)
            return jp.square(pred - y).mean()

        def body(carry, _):
            p, state, step_rng = carry
            step_rng, sample_rng = rax.split(step_rng)
            indices = rax.randint(sample_rng, shape=(batch_size,), minval=0, maxval=num_samples)
            x_batch = inputs[indices]
            y_batch = labels[indices]
            loss, grads = jax.value_and_grad(loss_fn)(p, x_batch, y_batch)
            updates, state = optimizer.update(grads, state, p)
            p = optax.apply_updates(p, updates)
            return (p, state, step_rng), loss

        (params, opt_state, rng), losses = jax.lax.scan(
            body,
            (params, opt_state, rng),
            xs=jp.arange(num_steps),
        )
        return params, opt_state, {'loss': losses.mean()}

    return train


def make_classifier_train_fn(model, optimizer, batch_size, num_steps):
    @jax.jit
    def train(params, opt_state, rng, inputs, labels):
        num_samples = inputs.shape[0]

        def loss_fn(p, x, y):
            logps = model.apply(p, x)
            losses = [
                -jp.take_along_axis(logp, y[:, factor_idx, None], axis=-1)[..., 0].mean()
                for factor_idx, logp in enumerate(logps)
            ]
            return jp.stack(losses).mean()

        def body(carry, _):
            p, state, step_rng = carry
            step_rng, sample_rng = rax.split(step_rng)
            indices = rax.randint(sample_rng, shape=(batch_size,), minval=0, maxval=num_samples)
            x_batch = inputs[indices]
            y_batch = labels[indices]
            loss, grads = jax.value_and_grad(loss_fn)(p, x_batch, y_batch)
            updates, state = optimizer.update(grads, state, p)
            p = optax.apply_updates(p, updates)
            return (p, state, step_rng), loss

        (params, opt_state, rng), losses = jax.lax.scan(
            body,
            (params, opt_state, rng),
            xs=jp.arange(num_steps),
        )
        return params, opt_state, {'loss': losses.mean()}

    return train


def make_update_policy_fn(
        optimizer,
        gamma,
        eps_clip,
        entropy_weight,
        num_epochs,
        clip_grad_norm,
        policy_anchor_kl_weight,
        player,
):
    @jax.jit
    def update(agent, opt_state, episodes, rewards, policy_anchor_agent, policy_anchor_active):
        data = build_ppo_training_data(episodes, rewards, gamma, player)

        def loss_fn(a):
            outputs = jax.vmap(lambda obs: a.call_seq({'obs_seq': obs}))(data['obs_seq'])
            logp_seq = outputs['logp_seq']
            value_seq = outputs['value_seq']
            new_logp_seq = jp.take_along_axis(
                logp_seq,
                data['action_seq'][..., None],
                axis=-1,
            )[..., 0]
            ratios = jp.exp(new_logp_seq - data['old_logp_seq'])
            advantages = data['return_seq'] - jax.lax.stop_gradient(value_seq)
            surrogate1 = ratios * advantages
            surrogate2 = jp.clip(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
            policy_loss = -jp.minimum(surrogate1, surrogate2).mean()
            value_loss = jp.square(value_seq - data['return_seq']).mean()
            entropy = categorical_entropy(logp_seq).mean()
            if policy_anchor_kl_weight > 0:
                anchor_outputs = jax.vmap(
                    lambda obs: policy_anchor_agent.call_seq({'obs_seq': obs})
                )(data['obs_seq'])
                anchor_logp_seq = jax.lax.stop_gradient(anchor_outputs['logp_seq'])
                anchor_probs = jp.exp(anchor_logp_seq)
                policy_anchor_kl = (anchor_probs * (anchor_logp_seq - logp_seq)).sum(axis=-1).mean()
                policy_anchor_loss = policy_anchor_active * policy_anchor_kl_weight * policy_anchor_kl
            else:
                policy_anchor_kl = jp.array(0.0)
                policy_anchor_loss = jp.array(0.0)
            total_loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy + policy_anchor_loss
            return total_loss, {
                'loss_ppo_policy': policy_loss,
                'loss_ppo_value': value_loss,
                'ppo_entropy': entropy,
                'policy_anchor_kl': policy_anchor_kl,
                'policy_anchor_loss': policy_anchor_loss,
            }

        def epoch_body(carry, _):
            a, state = carry
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(a)
            grad_norm = global_norm(grads)
            if clip_grad_norm is not None and clip_grad_norm > 0:
                grads = clip_grads_by_norm(grads, clip_grad_norm)
            updates, state = optimizer.update(grads, state, a)
            a = optax.apply_updates(a, updates)
            aux = {
                **aux,
                'loss_ppo_total': loss,
                'grad_ppo_norm': grad_norm,
            }
            return (a, state), aux

        (agent, opt_state), metrics = jax.lax.scan(
            epoch_body,
            (agent, opt_state),
            xs=jp.arange(num_epochs),
        )
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return agent, opt_state, metrics

    return update


def get_policy_anchor_config(hp):
    return hp['reciprocator'].get('policy_anchor', {})


def is_policy_anchor_enabled(hp):
    return bool(get_policy_anchor_config(hp).get('enabled', False))


def policy_anchor_scalar(value):
    return float(np.asarray(value))


def get_policy_anchor_score(metric_row, score_metric):
    p_same0 = float(metric_row['same_color_pickup_ratio_player0'])
    p_same1 = float(metric_row['same_color_pickup_ratio_player1'])
    if score_metric == 'min':
        return min(p_same0, p_same1)
    if score_metric == 'mean':
        return (p_same0 + p_same1) / 2
    if score_metric == 'agent0':
        return p_same0
    if score_metric == 'agent1':
        return p_same1
    raise ValueError(f'Unknown policy_anchor.score_metric {score_metric}')


def maybe_update_policy_anchor(state, metric_row, completed_timesteps, hp, anchor_agent0, anchor_agent1):
    anchor_hp = get_policy_anchor_config(hp)
    enabled = bool(anchor_hp.get('enabled', False))
    score_metric = str(anchor_hp.get('score_metric', 'min'))
    score = get_policy_anchor_score(metric_row, score_metric)
    best_score = policy_anchor_scalar(state['policy_anchor_best_score'])
    active = policy_anchor_scalar(state['policy_anchor_active'])
    did_update = False

    if enabled and completed_timesteps >= int(anchor_hp.get('start_timestep', 0)):
        min_p_same = float(anchor_hp.get('min_p_same', 0.0))
        improvement_margin = float(anchor_hp.get('improvement_margin', 0.0))
        improved = active <= 0 or score >= best_score + improvement_margin
        if score >= min_p_same and improved:
            state['policy_anchor_agent0'] = anchor_agent0
            state['policy_anchor_agent1'] = anchor_agent1
            state['policy_anchor_best_score'] = jp.array(score, dtype=jp.float32)
            state['policy_anchor_active'] = jp.array(1.0, dtype=jp.float32)
            best_score = score
            active = 1.0
            did_update = True

    return {
        'policy_anchor_active': active,
        'policy_anchor_score': score,
        'policy_anchor_best_score': best_score,
        'policy_anchor_update': float(did_update),
    }


def make_compute_reciprocal_reward_fn(
        scalar_model,
        transition_model,
        state_value_model,
        possible_states,
        transition_class_sizes,
        reward_type,
        normalize_reciprocal_reward,
):
    if reward_type not in RECIPROCAL_REWARD_TYPES:
        raise ValueError(f'Unknown reciprocal reward type {reward_type}')

    @jax.jit
    def compute(params, episodes, gamma):
        data = build_influence_training_data(episodes, gamma)
        batch_size, trace_length = episodes['rew'].shape[:2]
        full_reward = scalar_model.apply(params['reward'], data['full_inputs'])
        full_transition_logp = transition_model.apply(params['transition'], data['full_inputs'])
        cf_inputs = (data['cf_inputs0'], data['cf_inputs1'])
        state_values = state_value_model.apply(params['state_value'], possible_states)
        state_values = state_values.reshape((*transition_class_sizes, 2))

        def expected_transition_value(transition_logp):
            probs = tuple(jp.exp(factor_logp) for factor_logp in transition_logp)
            return jp.einsum(
                'na,nb,nc,abcp->np',
                probs[0],
                probs[1],
                probs[2],
                state_values,
                optimize='optimal',
            )

        full_transition_value = expected_transition_value(full_transition_logp)
        cf_rewards = tuple(
            scalar_model.apply(params[f'cf_reward{influencer_idx}'], cf_inputs[influencer_idx])
            for influencer_idx in range(2)
        )
        cf_transition_values = tuple(
            expected_transition_value(
                transition_model.apply(
                    params[f'cf_transition{influencer_idx}'],
                    cf_inputs[influencer_idx],
                )
            )
            for influencer_idx in range(2)
        )

        def voi(influencer_idx, influenced_idx):
            reward_influence = full_reward[:, influenced_idx] - cf_rewards[influencer_idx][:, influenced_idx]
            future_influence = (
                full_transition_value[:, influenced_idx]
                - cf_transition_values[influencer_idx][:, influenced_idx]
            )
            return (reward_influence + gamma * future_influence).reshape(batch_size, trace_length)

        voi_0_on_1 = voi(0, 1)
        voi_1_on_0 = voi(1, 0)
        if reward_type in ('grudge_minus_debit', 'signed_grudge_minus_debit'):
            voi_0_on_0 = voi(0, 0)
            voi_1_on_1 = voi(1, 1)
        else:
            voi_0_on_0 = jp.zeros_like(voi_0_on_1)
            voi_1_on_1 = jp.zeros_like(voi_1_on_0)

        def previous_discounted_sum(delta):
            def body(carry, delta_t):
                previous = carry
                current = gamma * carry + delta_t
                return current, previous

            _, values = jax.lax.scan(
                body,
                jp.zeros(delta.shape[0], dtype=delta.dtype),
                xs=delta.T,
            )
            return values.T

        def reciprocal_reward(self_voi_on_other, other_voi_on_self, other_voi_on_other):
            if reward_type in ('petty', 'signed_grudge'):
                grudge = previous_discounted_sum(other_voi_on_self)
                debit = jp.zeros_like(grudge)
            elif reward_type in (
                'petty_payoff',
                'signed_petty_payoff',
                'signed_grudge_petty_payoff',
            ):
                grudge = previous_discounted_sum(other_voi_on_self - self_voi_on_other)
                debit = jp.zeros_like(grudge)
            elif reward_type in ('grudge_minus_debit', 'signed_grudge_minus_debit'):
                grudge = previous_discounted_sum(other_voi_on_self)
                debit = previous_discounted_sum(self_voi_on_other + other_voi_on_other)
            else:
                raise ValueError(f'Unknown reciprocal reward type {reward_type}')

            if reward_type in ('signed_grudge', 'signed_petty_payoff'):
                reward = jp.sign(grudge) * self_voi_on_other
            elif reward_type == 'signed_grudge_petty_payoff':
                reward = grudge * jp.sign(self_voi_on_other)
            elif reward_type == 'signed_grudge_minus_debit':
                reward = jp.sign(grudge - debit) * self_voi_on_other
            elif reward_type == 'grudge_minus_debit':
                reward = (grudge - debit) * self_voi_on_other
            else:
                reward = grudge * self_voi_on_other
            return reward, grudge

        reward0, grudge0 = reciprocal_reward(voi_0_on_1, voi_1_on_0, voi_1_on_1)
        reward1, grudge1 = reciprocal_reward(voi_1_on_0, voi_0_on_1, voi_0_on_0)

        if normalize_reciprocal_reward:
            reward0 = (reward0 - reward0.mean()) / (reward0.std() + 1e-7)
            reward1 = (reward1 - reward1.mean()) / (reward1.std() + 1e-7)

        reciprocal_rewards = jp.stack([reward0, reward1], axis=-1)
        return {
            'reciprocal_rewards': reciprocal_rewards,
            'mean_reciprocal_reward_player0': reward0.mean(),
            'mean_reciprocal_reward_player1': reward1.mean(),
            'mean_cumulative_reciprocal_reward_player0': reward0.sum(axis=1).mean(),
            'mean_cumulative_reciprocal_reward_player1': reward1.sum(axis=1).mean(),
            'mean_end_grudge_player0': grudge0[:, -1].mean(),
            'mean_end_grudge_player1': grudge1[:, -1].mean(),
            'mean_voi_0_on_1': voi_0_on_1.mean(),
            'mean_voi_1_on_0': voi_1_on_0.mean(),
        }

    return compute


@jax.jit
def compute_reciprocator_diagnostics(episodes, reciprocal_rewards, total_rewards):
    env_rewards = episodes['rew']
    batch_size, trace_length = env_rewards.shape[:2]
    coin_owner = episodes['coin_owner'][:, :-1, 0]
    coin_pos = episodes['coin_pos'][:, :-1]
    player0_pos = episodes['player1_pos'][:, 1:]
    player1_pos = episodes['player2_pos'][:, 1:]
    actions = episodes['act']

    player0_takes = (player0_pos == coin_pos).all(axis=-1)
    player1_takes = (player1_pos == coin_pos).all(axis=-1)
    player0_same = player0_takes & (coin_owner == 0)
    player1_same = player1_takes & (coin_owner == 1)
    player0_other = player0_takes & (coin_owner == 1)
    player1_other = player1_takes & (coin_owner == 0)

    action_one_hot = jax.nn.one_hot(actions, 4).mean(axis=(0, 1))
    return {
        'std_reciprocal_reward_player0': reciprocal_rewards[:, :, 0].std(),
        'std_reciprocal_reward_player1': reciprocal_rewards[:, :, 1].std(),
        'mean_abs_reciprocal_reward_player0': jp.abs(reciprocal_rewards[:, :, 0]).mean(),
        'mean_abs_reciprocal_reward_player1': jp.abs(reciprocal_rewards[:, :, 1]).mean(),
        'avg_env_return_player0': env_rewards[:, :, 0].sum(axis=1).mean(),
        'avg_env_return_player1': env_rewards[:, :, 1].sum(axis=1).mean(),
        'avg_shaped_return_player0': total_rewards[:, :, 0].sum(axis=1).mean(),
        'avg_shaped_return_player1': total_rewards[:, :, 1].sum(axis=1).mean(),
        'corr_reciprocal_reward_same_color_pickup_player0': safe_corr(
            reciprocal_rewards[:, :, 0], player0_same.astype(reciprocal_rewards.dtype)
        ),
        'corr_reciprocal_reward_same_color_pickup_player1': safe_corr(
            reciprocal_rewards[:, :, 1], player1_same.astype(reciprocal_rewards.dtype)
        ),
        'corr_reciprocal_reward_other_color_pickup_player0': safe_corr(
            reciprocal_rewards[:, :, 0], player0_other.astype(reciprocal_rewards.dtype)
        ),
        'corr_reciprocal_reward_other_color_pickup_player1': safe_corr(
            reciprocal_rewards[:, :, 1], player1_other.astype(reciprocal_rewards.dtype)
        ),
        'action_frequency_left_player0': action_one_hot[0, 0],
        'action_frequency_right_player0': action_one_hot[0, 1],
        'action_frequency_up_player0': action_one_hot[0, 2],
        'action_frequency_down_player0': action_one_hot[0, 3],
        'action_frequency_left_player1': action_one_hot[1, 0],
        'action_frequency_right_player1': action_one_hot[1, 1],
        'action_frequency_up_player1': action_one_hot[1, 2],
        'action_frequency_down_player1': action_one_hot[1, 3],
    }


def init_reciprocator_metrics_csv(save_path, hp):
    csv_path = os.path.join(save_path, get_metrics_csv_filename(hp))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RECIPROCAL_CSV_FIELDNAMES)
        writer.writeheader()
    return csv_path


def append_reciprocator_metric_row(csv_path, row):
    full_row = {field: row.get(field, '') for field in RECIPROCAL_CSV_FIELDNAMES}
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RECIPROCAL_CSV_FIELDNAMES)
        writer.writerow(full_row)


def make_possible_single_coin_states(obs_shape):
    _, height, width = obs_shape
    num_position_classes = height * width
    coin_classes = 2 * num_position_classes
    labels = jp.stack(
        jp.meshgrid(
            jp.arange(num_position_classes),
            jp.arange(num_position_classes),
            jp.arange(coin_classes),
            indexing='ij',
        ),
        axis=-1,
    ).reshape(-1, 3)
    player_channels = jax.nn.one_hot(labels[:, :2], num_position_classes)
    coin_channels = jax.nn.one_hot(labels[:, 2], coin_classes).reshape(-1, 2, num_position_classes)
    return jp.concatenate([player_channels, coin_channels], axis=1).reshape(-1, 4, height, width)


def initialize_value_of_influence(rng, obs_shape, hp):
    influence_hp = hp['reciprocator']['influence']
    hidden_size = int(influence_hp['hidden_size'])
    _, height, width = obs_shape
    num_position_classes = height * width
    transition_class_sizes = (
        num_position_classes,
        num_position_classes,
        2 * num_position_classes,
    )
    scalar_model = ScalarPredictor(hidden_size=hidden_size, output_size=2)
    transition_model = DiscreteTransitionPredictor(
        hidden_size=hidden_size,
        num_classes_by_factor=transition_class_sizes,
    )
    state_value_model = CoinStateValuePredictor(
        obs_shape=obs_shape,
        hidden_size=hidden_size,
        out_channels=max(1, hidden_size // 4),
        output_size=2,
    )
    obs_dim = int(np.prod(obs_shape))
    full_input_dim = 2 + obs_dim + 1
    cf_input_dim = 1 + obs_dim + 1

    rngs = rax.split(rng, 8)
    dummy_state = jp.zeros((1, obs_dim), dtype=jp.float32)
    dummy_full = jp.zeros((1, full_input_dim), dtype=jp.float32)
    dummy_cf = jp.zeros((1, cf_input_dim), dtype=jp.float32)
    params = {
        'state_value': state_value_model.init(rngs[0], dummy_state),
        'reward': scalar_model.init(rngs[1], dummy_full),
        'transition': transition_model.init(rngs[2], dummy_full),
        'cf_reward0': scalar_model.init(rngs[3], dummy_cf),
        'cf_reward1': scalar_model.init(rngs[4], dummy_cf),
        'cf_transition0': transition_model.init(rngs[5], dummy_cf),
        'cf_transition1': transition_model.init(rngs[6], dummy_cf),
    }

    optimizer = optax.adam(float(influence_hp['lr']))
    opt_states = {name: optimizer.init(param) for name, param in params.items()}
    num_steps = int(influence_hp['target_epochs']) * int(influence_hp['num_train_batches'])
    scalar_train_fn = make_predictor_train_fn(
        scalar_model,
        optimizer,
        int(influence_hp['target_batch_size']),
        num_steps,
    )
    state_value_train_fn = make_predictor_train_fn(
        state_value_model,
        optimizer,
        int(influence_hp['target_batch_size']),
        num_steps,
    )
    transition_train_fn = make_classifier_train_fn(
        transition_model,
        optimizer,
        int(influence_hp['target_batch_size']),
        num_steps,
    )
    compute_reciprocal_reward = make_compute_reciprocal_reward_fn(
        scalar_model,
        transition_model,
        state_value_model,
        make_possible_single_coin_states(obs_shape),
        transition_class_sizes,
        hp['reciprocator']['reciprocal_reward_type'],
        bool(hp['reciprocator']['normalize_reciprocal_reward']),
    )
    return {
        'params': params,
        'opt_states': opt_states,
        'optimizer': optimizer,
        'scalar_train_fn': scalar_train_fn,
        'state_value_train_fn': state_value_train_fn,
        'transition_train_fn': transition_train_fn,
        'compute_reciprocal_reward': compute_reciprocal_reward,
    }


def train_value_of_influence(voi_state, episodes, rng, gamma):
    data = build_influence_training_data(episodes, gamma)
    params = dict(voi_state['params'])
    opt_states = dict(voi_state['opt_states'])
    rngs = rax.split(rng, 7)

    params['state_value'], opt_states['state_value'], state_value_loss = voi_state['state_value_train_fn'](
        params['state_value'],
        opt_states['state_value'],
        rngs[0],
        data['state_inputs'],
        data['state_value_labels'],
    )
    params['reward'], opt_states['reward'], reward_loss = voi_state['scalar_train_fn'](
        params['reward'],
        opt_states['reward'],
        rngs[1],
        data['full_inputs'],
        data['reward_labels'],
    )
    params['transition'], opt_states['transition'], transition_loss = voi_state['transition_train_fn'](
        params['transition'],
        opt_states['transition'],
        rngs[2],
        data['transition_full_inputs'],
        data['transition_labels'],
    )
    params['cf_reward0'], opt_states['cf_reward0'], cf_reward0_loss = voi_state['scalar_train_fn'](
        params['cf_reward0'],
        opt_states['cf_reward0'],
        rngs[3],
        data['cf_inputs0'],
        data['reward_labels'],
    )
    params['cf_reward1'], opt_states['cf_reward1'], cf_reward1_loss = voi_state['scalar_train_fn'](
        params['cf_reward1'],
        opt_states['cf_reward1'],
        rngs[4],
        data['cf_inputs1'],
        data['reward_labels'],
    )
    params['cf_transition0'], opt_states['cf_transition0'], cf_transition0_loss = voi_state['transition_train_fn'](
        params['cf_transition0'],
        opt_states['cf_transition0'],
        rngs[5],
        data['transition_cf_inputs0'],
        data['transition_labels'],
    )
    params['cf_transition1'], opt_states['cf_transition1'], cf_transition1_loss = voi_state['transition_train_fn'](
        params['cf_transition1'],
        opt_states['cf_transition1'],
        rngs[6],
        data['transition_cf_inputs1'],
        data['transition_labels'],
    )

    voi_state['params'] = params
    voi_state['opt_states'] = opt_states
    metrics = {
        'loss_voi_state_value': state_value_loss['loss'],
        'loss_voi_reward': reward_loss['loss'],
        'loss_voi_transition': transition_loss['loss'],
        'loss_voi_cf_reward0': cf_reward0_loss['loss'],
        'loss_voi_cf_reward1': cf_reward1_loss['loss'],
        'loss_voi_cf_transition0': cf_transition0_loss['loss'],
        'loss_voi_cf_transition1': cf_transition1_loss['loss'],
    }
    return metrics


def set_up_state_from_config(hp):
    dummy_env, _ = CoinGame.init(
        rng=rax.PRNGKey(hp['seed']),
        **coin_game_params(hp),
    )
    dummy_episode = make_zero_episode(trace_length=hp['game']['game_length'], coin_game=dummy_env)
    dummy_obs_seq = dummy_episode['obs'][:, 0].reshape(dummy_episode['obs'].shape[0], -1)
    rng, agent0_rng, agent1_rng, voi_rng = rax.split(rax.PRNGKey(hp['seed']), 4)

    policy_hp = hp['reciprocator']['policy']
    policy_anchor_hp = get_policy_anchor_config(hp)
    policy_anchor_kl_weight = (
        float(policy_anchor_hp.get('kl_weight', 0.0))
        if bool(policy_anchor_hp.get('enabled', False))
        else 0.0
    )
    policy_model = policy_hp.get('model', 'conv_gru')
    if policy_model == 'conv_gru':
        agent_module = ConvGRUActorCriticCoinAgent(
            num_actions=dummy_env.NUM_ACTIONS,
            obs_shape=tuple(dummy_env.OBS_SHAPE),
            hidden_size_actor=int(policy_hp['hidden_size_actor']),
            hidden_size_value=int(policy_hp['hidden_size_value']),
            layers_before_gru_actor=int(policy_hp['layers_before_gru_actor']),
            layers_before_gru_value=int(policy_hp['layers_before_gru_value']),
            conv_out_channels=int(policy_hp['conv_out_channels']),
        )
    elif policy_model == 'gru':
        agent_module = GRUActorCriticCoinAgent(
            num_actions=dummy_env.NUM_ACTIONS,
            hidden_size_actor=int(policy_hp['hidden_size_actor']),
            hidden_size_value=int(policy_hp['hidden_size_value']),
            layers_before_gru_actor=int(policy_hp['layers_before_gru_actor']),
            layers_before_gru_value=int(policy_hp['layers_before_gru_value']),
        )
    else:
        raise ValueError(f'Unknown Reciprocator policy model {policy_model}')
    agent0_params = agent_module.init(agent0_rng, {'obs_seq': dummy_obs_seq, 'rng': rax.PRNGKey(0), 't': 0})
    agent1_params = agent_module.init(agent1_rng, {'obs_seq': dummy_obs_seq, 'rng': rax.PRNGKey(1), 't': 0})
    agent0 = CoinAgent(params=agent0_params, model=agent_module, player=0)
    agent1 = CoinAgent(params=agent1_params, model=agent_module, player=1)
    policy_optimizer = optax.adam(float(policy_hp['lr']))
    agent0_opt = Optimizer(policy_optimizer, policy_optimizer.init(agent0))
    agent1_opt = Optimizer(policy_optimizer, policy_optimizer.init(agent1))

    carries0 = agent0.get_initial_carries()
    carries1 = agent1.get_initial_carries()
    carries = {
        'c_0_actor': carries0['carry_actor'],
        'c_0_qvalue': carries0['carry_qvalue'],
        'c_1_actor': carries1['carry_actor'],
        'c_1_qvalue': carries1['carry_qvalue'],
    }
    voi_state = initialize_value_of_influence(
        voi_rng,
        tuple(dummy_env.OBS_SHAPE),
        hp,
    )
    update_policy0 = make_update_policy_fn(
        policy_optimizer,
        float(hp['reward_discount']),
        float(policy_hp['eps_clip']),
        float(policy_hp['entropy_weight']),
        int(policy_hp['ppo_epochs']),
        float(policy_hp['clip_grad_norm']) if 'clip_grad_norm' in policy_hp else None,
        policy_anchor_kl_weight,
        player=0,
    )
    update_policy1 = make_update_policy_fn(
        policy_optimizer,
        float(hp['reward_discount']),
        float(policy_hp['eps_clip']),
        float(policy_hp['entropy_weight']),
        int(policy_hp['ppo_epochs']),
        float(policy_hp['clip_grad_norm']) if 'clip_grad_norm' in policy_hp else None,
        policy_anchor_kl_weight,
        player=1,
    )
    state = {
        'rng': rng,
        'agent0': agent0,
        'agent1': agent1,
        'agent0_opt': agent0_opt,
        'agent1_opt': agent1_opt,
        'voi': voi_state,
        'update_policy0': update_policy0,
        'update_policy1': update_policy1,
        'policy_anchor_agent0': agent0,
        'policy_anchor_agent1': agent1,
        'policy_anchor_active': jp.array(0.0, dtype=jp.float32),
        'policy_anchor_best_score': jp.array(-1.0, dtype=jp.float32),
    }
    return state, carries


def should_update_influence(iteration, hp):
    influence_hp = hp['reciprocator']['influence']
    init_iterations = int(influence_hp['num_initialization_iterations'])
    iteration_count = iteration + 1
    if iteration_count <= init_iterations:
        period = int(influence_hp['initialization_target_period'])
    else:
        period = int(influence_hp['target_period'])
    return period > 0 and iteration_count % period == 0


def should_update_policy(iteration, hp):
    init_iterations = int(hp['reciprocator']['influence']['num_initialization_iterations'])
    return iteration + 1 > init_iterations


def validate_reciprocator_config(hp):
    if not hp['just_self_play']:
        raise ValueError('reciprocator_coin_train.py currently implements Reciprocator self-play only.')
    if hp['agent_0'] != 'reciprocator' or hp['agent_1'] != 'reciprocator':
        raise ValueError('Set hp.agent_0=reciprocator and hp.agent_1=reciprocator.')
    if hp['game']['height'] != 3 or hp['game']['width'] != 3:
        raise ValueError('This Reciprocator entry is intended for the same 3x3 Coin Game setting as LOQA self-play.')
    if hp['game']['game_length'] != 200:
        raise ValueError('This Reciprocator entry expects 200 steps per episode, matching the LOQA self-play setup.')
    if int(hp['seed']) < 0:
        raise ValueError('hp.seed must be a non-negative integer.')
    policy_anchor_hp = hp['reciprocator'].get('policy_anchor', {})
    if bool(policy_anchor_hp.get('enabled', False)):
        if str(policy_anchor_hp.get('score_metric', 'min')) not in ('min', 'mean', 'agent0', 'agent1'):
            raise ValueError('hp.reciprocator.policy_anchor.score_metric must be one of min, mean, agent0, agent1.')
        if not 0 <= float(policy_anchor_hp.get('min_p_same', 0.0)) <= 1:
            raise ValueError('hp.reciprocator.policy_anchor.min_p_same must be in [0, 1].')
        if float(policy_anchor_hp.get('improvement_margin', 0.0)) < 0:
            raise ValueError('hp.reciprocator.policy_anchor.improvement_margin must be non-negative.')
        if float(policy_anchor_hp.get('kl_weight', 0.0)) <= 0:
            raise ValueError('hp.reciprocator.policy_anchor.kl_weight must be positive when enabled.')
        if int(policy_anchor_hp.get('start_timestep', 0)) < 0:
            raise ValueError('hp.reciprocator.policy_anchor.start_timestep must be non-negative.')
    if get_total_training_timesteps(hp) == 30_000_000 and get_expected_metrics_csv_rows(hp) != 7500:
        raise ValueError('A 3e7-timestep Reciprocator run must be configured to write exactly 7500 CSV rows.')


def get_expected_metrics_csv_rows(hp):
    num_training_iterations = get_num_training_iterations(hp)
    timesteps_per_iteration = get_num_training_timesteps_per_iteration(hp)
    metrics_log_timestep_freq = get_metrics_log_timestep_freq(hp)
    iterations_per_row = (metrics_log_timestep_freq + timesteps_per_iteration - 1) // timesteps_per_iteration
    return (num_training_iterations + iterations_per_row - 1) // iterations_per_row


def slim_episode_for_influence(episodes):
    return {
        key: episodes[key]
        for key in ('obs', 'act', 'rew', 'coin_owner', 'coin_pos', 'player1_pos', 'player2_pos')
    }


def train(hp, log_wandb):
    validate_reciprocator_config(hp)
    run_id = wandb.run.id if log_wandb else datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join(hp['save_dir'], run_id)
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()

    hp = flax.core.FrozenDict(hp)
    num_training_iterations = get_num_training_iterations(hp)
    total_training_episodes = get_total_training_episodes(hp)
    episodes_per_iteration = get_num_training_episodes_per_iteration(hp)
    timesteps_per_iteration = get_num_training_timesteps_per_iteration(hp)
    episode_length = int(hp['game']['game_length'])
    total_training_timesteps = get_total_training_timesteps(hp)
    metrics_log_timestep_freq = get_metrics_log_timestep_freq(hp)
    state, carries = set_up_state_from_config(hp)
    print('****reciprocator self play (jax)****')
    print(f'num training iterations: {num_training_iterations}')
    print(f'episodes per iteration: {episodes_per_iteration}')
    print(f'episode length: {episode_length} timesteps')
    print(
        f'timesteps per iteration/window: {timesteps_per_iteration} '
        f'({episodes_per_iteration} episodes x {episode_length} timesteps)'
    )
    print(f'total training episodes: {total_training_episodes}')
    print(f'total training timesteps: {total_training_timesteps}')
    print(f'metrics log timestep frequency: {metrics_log_timestep_freq}')
    print(f'expected metrics csv rows: {get_expected_metrics_csv_rows(hp)}')

    iteration_metrics_csv_path = init_reciprocator_metrics_csv(save_path, hp)
    print(f'metrics csv path: {iteration_metrics_csv_path}')

    dummy_env, _ = CoinGame.init(
        rng=rax.PRNGKey(hp['seed']),
        **coin_game_params(hp),
    )
    episode_stats_jitted = jax.jit(lambda es: episode_stats(es, dummy_env))
    target_buffer = deque(maxlen=int(hp['reciprocator']['influence']['target_buffer_size']))

    metric_window_batches = []
    metric_window_update_rows = []
    window_start_iteration = 0
    window_start_episode = 0
    window_start_timestep = 0
    completed_episodes = 0
    completed_timesteps = 0
    last_metric_timestep = 0

    for i in range(num_training_iterations):
        state['rng'], rng = rax.split(state['rng'])
        metric_anchor_agent0 = state['agent0']
        metric_anchor_agent1 = state['agent1']
        episodes = generate_selfplay_episodes(state['agent0'], state['agent1'], carries, rng, hp)
        target_buffer.append(slim_episode_for_influence(episodes))
        metric_window_batches.append(episodes)

        state['rng'], influence_rng = rax.split(state['rng'])
        update_metric_row = {}
        if should_update_influence(i, hp):
            target_episodes = tree_concatenate(list(target_buffer))
            influence_metrics = train_value_of_influence(
                state['voi'],
                target_episodes,
                influence_rng,
                float(hp['reward_discount']),
            )
            update_metric_row.update({key: scalar_mean(value) for key, value in influence_metrics.items()})

        reciprocal_out = state['voi']['compute_reciprocal_reward'](
            state['voi']['params'],
            episodes,
            float(hp['reward_discount']),
        )
        reciprocal_rewards = reciprocal_out['reciprocal_rewards']
        total_rewards = episodes['rew'] + float(hp['reciprocator']['reciprocal_reward_weight']) * reciprocal_rewards
        update_metric_row.update({
            key: scalar_mean(value)
            for key, value in reciprocal_out.items()
            if key != 'reciprocal_rewards'
        })
        diagnostic_metrics = compute_reciprocator_diagnostics(episodes, reciprocal_rewards, total_rewards)
        update_metric_row.update({key: scalar_mean(value) for key, value in diagnostic_metrics.items()})

        if should_update_policy(i, hp):
            new_agent0, new_opt_state0, ppo_metrics0 = state['update_policy0'](
                state['agent0'],
                state['agent0_opt'].opt_state,
                episodes,
                total_rewards,
                state['policy_anchor_agent0'],
                state['policy_anchor_active'],
            )
            new_agent1, new_opt_state1, ppo_metrics1 = state['update_policy1'](
                state['agent1'],
                state['agent1_opt'].opt_state,
                episodes,
                total_rewards,
                state['policy_anchor_agent1'],
                state['policy_anchor_active'],
            )
            state['agent0'] = new_agent0
            state['agent1'] = new_agent1
            state['agent0_opt'] = state['agent0_opt'].replace(opt_state=new_opt_state0)
            state['agent1_opt'] = state['agent1_opt'].replace(opt_state=new_opt_state1)
            ppo_metrics = jax.tree_util.tree_map(lambda x, y: (x + y) / 2, ppo_metrics0, ppo_metrics1)
            update_metric_row.update({key: scalar_mean(value) for key, value in ppo_metrics.items()})

        completed_episodes += episodes_per_iteration
        completed_timesteps += timesteps_per_iteration
        update_metric_row['walltime'] = time.time() - start_time
        metric_window_update_rows.append(update_metric_row)

        should_log_metrics = (
            completed_timesteps - last_metric_timestep >= metrics_log_timestep_freq
            or i == num_training_iterations - 1
        )
        if should_log_metrics:
            metric_window_episodes = tree_concatenate(metric_window_batches)
            metric_window_stats = episode_stats_jitted(metric_window_episodes)
            metric_row = compute_iteration_metric_row(
                episode_batches=metric_window_batches,
                iteration=i,
                episode=completed_episodes,
                timestep=completed_timesteps,
                window_start_iteration=window_start_iteration,
                window_start_episode=window_start_episode,
                window_start_timestep=window_start_timestep,
                statistics=metric_window_stats,
            )
            for key in RECIPROCAL_METRIC_FIELDNAMES + ['walltime']:
                values = [row[key] for row in metric_window_update_rows if key in row]
                if values:
                    metric_row[key] = float(np.mean(values))
            metric_row.update(
                maybe_update_policy_anchor(
                    state,
                    metric_row,
                    completed_timesteps,
                    hp,
                    metric_anchor_agent0,
                    metric_anchor_agent1,
                )
            )

            append_reciprocator_metric_row(iteration_metrics_csv_path, metric_row)
            if log_wandb:
                wandb.log(format_wandb_metrics(metric_row), step=completed_timesteps)

            metric_window_batches = []
            metric_window_update_rows = []
            last_metric_timestep = completed_timesteps
            window_start_iteration = i + 1
            window_start_episode = completed_episodes
            window_start_timestep = completed_timesteps

        if i % hp['eval_every'] == 0:
            print(f'iteration {i}, timestep {completed_timesteps}')
            print({
                key: round(value, 4)
                for key, value in update_metric_row.items()
                if isinstance(value, float)
            })
            stats = episode_stats_jitted(episodes)
            print(stats)

        if i % hp['save_every'] == 0:
            minimal_state = {
                'agent0': npify(state['agent0']),
                'agent1': npify(state['agent1']),
                'policy_anchor_agent0': npify(state['policy_anchor_agent0']),
                'policy_anchor_agent1': npify(state['policy_anchor_agent1']),
                'policy_anchor_active': npify(state['policy_anchor_active']),
                'policy_anchor_best_score': npify(state['policy_anchor_best_score']),
                'voi_params': npify(state['voi']['params']),
                'hp': hp,
            }
            with open(os.path.join(save_path, f'minimal_state_{i}'), 'wb') as f:
                pickle.dump(flax.serialization.to_state_dict(minimal_state), f)


@hydra.main(version_base=None, config_path='conf/coin_conf', config_name='coin_config')
def main(cfg: DictConfig) -> None:
    jp.set_printoptions(precision=3)
    config.update('jax_disable_jit', cfg.jax.jax_disable_jit)
    config.update('jax_debug_nans', cfg.jax.jax_debug_nans)
    hp = OmegaConf.to_container(cfg.hp, resolve=True)
    print(OmegaConf.to_yaml(cfg.hp))

    log_wandb = cfg.wandb.state == 'enabled'
    if log_wandb:
        initialize_wandb(cfg.wandb, hp)

    train(hp, log_wandb)


if __name__ == '__main__':
    main()
