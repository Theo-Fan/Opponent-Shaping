import csv
import os
import pickle
import shutil
import time
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
from coin_game import MOVES, CoinGame, coin_game_params
from coin_train import (
    Optimizer,
    TRAIN_ITERATION_METRICS_FIELDNAMES,
    format_wandb_metrics,
    get_metrics_csv_filename,
    get_metrics_log_timestep_freq,
    get_num_training_episodes_per_iteration,
    get_num_training_iterations,
    get_num_training_timesteps_per_iteration,
    get_total_training_episodes,
    get_total_training_timesteps,
    scalar_mean,
)
from mfos_coin_agent import MFOSCoinAgent
from utils import clip_grads_by_norm, global_norm, npify, rscope, slurm_infos


MFOS_METRIC_FIELDNAMES = [
    'loss_mfos_total',
    'loss_mfos_policy',
    'loss_mfos_value',
    'mfos_entropy',
    'grad_mfos_norm',
    'theta_mean',
    'theta_std',
    'action_frequency_left_player0',
    'action_frequency_right_player0',
    'action_frequency_up_player0',
    'action_frequency_down_player0',
    'action_frequency_left_player1',
    'action_frequency_right_player1',
    'action_frequency_up_player1',
    'action_frequency_down_player1',
]
MFOS_CSV_FIELDNAMES = [
    *TRAIN_ITERATION_METRICS_FIELDNAMES,
    *[field for field in MFOS_METRIC_FIELDNAMES if field not in TRAIN_ITERATION_METRICS_FIELDNAMES],
]


def categorical_entropy(logp):
    return -jp.sum(jp.nan_to_num(jp.exp(logp) * logp), axis=-1)


def init_mfos_metrics_csv(save_path, hp):
    csv_path = os.path.join(save_path, get_metrics_csv_filename(hp))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MFOS_CSV_FIELDNAMES)
        writer.writeheader()
    return csv_path


def append_mfos_metric_row(csv_path, row):
    full_row = {field: row.get(field, '') for field in MFOS_CSV_FIELDNAMES}
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MFOS_CSV_FIELDNAMES)
        writer.writerow(full_row)


def make_mfos_state(obs, own_previous_reward, other_previous_reward, done, height, width):
    own_reward_channel = jp.ones((1, height, width), dtype=obs.dtype) * own_previous_reward
    other_reward_channel = jp.ones((1, height, width), dtype=obs.dtype) * other_previous_reward
    done_channel = jp.ones((1, height, width), dtype=obs.dtype) * done
    return jp.concatenate([obs, own_reward_channel, other_reward_channel, done_channel], axis=0)


def get_mfos_inner_episode_length(hp):
    return int(hp['mfos'].get('inner_episode_length', hp['game']['game_length']))


def batch_initial_carry(carry, batch_size):
    return jax.tree_map(lambda x: jp.repeat(x[None], batch_size, axis=0), carry)


@partial(jax.jit, static_argnames=('hp', 'mfos_input_shape'))
def generate_mfos_episodes(agent0, agent1, rng, hp, mfos_input_shape):
    batch_size = hp['batch_size']
    metric_episode_length = hp['game']['game_length']
    inner_episode_length = get_mfos_inner_episode_length(hp)
    num_inner_episodes = metric_episode_length // inner_episode_length
    height = hp['game']['height']
    width = hp['game']['width']
    state_dtype = jp.float32
    theta0_init = jp.repeat(agent0.get_initial_theta()[None], batch_size, axis=0)
    theta1_init = jp.repeat(agent1.get_initial_theta()[None], batch_size, axis=0)

    def run_inner_episode(carry, _):
        rng, theta0, theta1 = carry
        rng, env_rng, play_rng = rax.split(rng, 3)
        env_rngs = rax.split(env_rng, batch_size)
        env = jax.vmap(lambda r: CoinGame.init(rng=r, **coin_game_params(hp))[0])(env_rngs)
        obs = jax.vmap(lambda e: e.get_obs())(env)
        previous_rewards = jp.zeros((batch_size, 2), dtype=state_dtype)
        carries0 = agent0.get_initial_carries()
        carries1 = agent1.get_initial_carries()
        c0_actor = batch_initial_carry(carries0['carry_actor'], batch_size)
        c0_value = batch_initial_carry(carries0['carry_value'], batch_size)
        c1_actor = batch_initial_carry(carries1['carry_actor'], batch_size)
        c1_value = batch_initial_carry(carries1['carry_value'], batch_size)

        def step_fn(step_carry, _):
            env, obs, rng, previous_rewards, c0_actor, c0_value, c1_actor, c1_value = step_carry
            rng, rng0, rng1 = rax.split(rng, 3)
            rng0 = rax.split(rng0, batch_size)
            rng1 = rax.split(rng1, batch_size)
            state0 = jax.vmap(make_mfos_state, in_axes=(0, 0, 0, None, None, None))(
                obs[:, 0],
                previous_rewards[:, 0],
                previous_rewards[:, 1],
                0.,
                height,
                width,
            )
            state1 = jax.vmap(make_mfos_state, in_axes=(0, 0, 0, None, None, None))(
                obs[:, 1],
                previous_rewards[:, 1],
                previous_rewards[:, 0],
                0.,
                height,
                width,
            )
            out0 = jax.vmap(
                lambda s, th, ca, cv, r: agent0.call_step({
                    'state': s.reshape(-1),
                    'theta': th,
                    'carry_actor': ca,
                    'carry_value': cv,
                    'rng': r,
                })
            )(state0, theta0, c0_actor, c0_value, rng0)
            out1 = jax.vmap(
                lambda s, th, ca, cv, r: agent1.call_step({
                    'state': s.reshape(-1),
                    'theta': th,
                    'carry_actor': ca,
                    'carry_value': cv,
                    'rng': r,
                })
            )(state1, theta1, c1_actor, c1_value, rng1)
            actions = jp.stack([out0['action'], out1['action']], axis=1)
            logp = jp.stack([out0['logp'], out1['logp']], axis=1)
            obs_before = obs
            coin_pos_before = env.coin_pos
            coin_owner_before = env.coin_owner
            player1_pos_before = env.players_pos[:, 0]
            player2_pos_before = env.players_pos[:, 1]

            next_players_pos = (
                env.players_pos + MOVES[actions]
            ) % jp.array([height, width], dtype=env.players_pos.dtype)
            takers = (next_players_pos == env.coin_pos[:, None, :]).all(axis=-1)
            coin_owner = env.coin_owner
            same_pickup = takers & (coin_owner[:, None] == jp.arange(2)[None, :])
            different_pickup = takers & (coin_owner[:, None] != jp.arange(2)[None, :])

            env, obs, rewards = jax.vmap(lambda e, a: e.step(a))(env, actions)
            rewards = rewards.astype(state_dtype)
            step_data = {
                'obs_before': obs_before,
                'coin_pos_before': coin_pos_before,
                'coin_owner_before': coin_owner_before,
                'player1_pos_before': player1_pos_before,
                'player2_pos_before': player2_pos_before,
                'mfos_state': jp.stack([state0, state1], axis=1),
                'act': actions,
                'logp': logp,
                'rew': rewards,
                'same_pickup': same_pickup.astype(state_dtype),
                'different_pickup': different_pickup.astype(state_dtype),
            }
            return (
                env,
                obs,
                rng,
                rewards,
                out0['carry_actor'],
                out0['carry_value'],
                out1['carry_actor'],
                out1['carry_value'],
            ), step_data

        init = (env, obs, play_rng, previous_rewards, c0_actor, c0_value, c1_actor, c1_value)
        (_, _, rng, _, _, _, _, _), chunk = jax.lax.scan(
            step_fn,
            init,
            xs=(),
            length=inner_episode_length,
        )
        chunk = jax.tree_map(lambda x: jp.swapaxes(x, 0, 1), chunk)
        theta0_next = agent0.theta_from_batch_seq({
            'state_seq': chunk['mfos_state'][:, :, 0].reshape(batch_size, inner_episode_length, -1),
        })
        theta1_next = agent1.theta_from_batch_seq({
            'state_seq': chunk['mfos_state'][:, :, 1].reshape(batch_size, inner_episode_length, -1),
        })
        theta_metrics = jp.stack([theta0_next, theta1_next], axis=1)
        return (rng, theta0_next, theta1_next), {'chunk': chunk, 'theta': theta_metrics}

    (_, _, _), outs = jax.lax.scan(
        run_inner_episode,
        (rng, theta0_init, theta1_init),
        xs=(),
        length=num_inner_episodes,
    )
    chunks = jax.tree_map(lambda x: jp.swapaxes(x, 0, 1), outs['chunk'])
    theta_metrics = jp.swapaxes(outs['theta'], 0, 1)

    def flatten_time(x):
        return x.reshape(batch_size, num_inner_episodes * inner_episode_length, *x.shape[3:])

    flat = jax.tree_map(flatten_time, chunks)
    obs = jp.concatenate(
        [
            flat['obs_before'],
            flat['obs_before'][:, -1:, :],
        ],
        axis=1,
    )
    coin_pos = jp.concatenate([flat['coin_pos_before'], flat['coin_pos_before'][:, -1:]], axis=1)
    coin_owner = jp.concatenate([flat['coin_owner_before'], flat['coin_owner_before'][:, -1:]], axis=1)[..., None]
    player1_pos = jp.concatenate([flat['player1_pos_before'], flat['player1_pos_before'][:, -1:]], axis=1)
    player2_pos = jp.concatenate([flat['player2_pos_before'], flat['player2_pos_before'][:, -1:]], axis=1)
    episodes = {
        'obs': obs,
        'act': flat['act'],
        'rew': flat['rew'],
        'coin_pos': coin_pos,
        'coin_owner': coin_owner,
        'player1_pos': player1_pos,
        'player2_pos': player2_pos,
        'logp': flat['logp'],
        'mfos_state': flat['mfos_state'],
        'same_pickup': flat['same_pickup'],
        'different_pickup': flat['different_pickup'],
    }
    return episodes, theta_metrics


def mfos_discounted_returns(rewards, gamma, inner_episode_length):
    batch_size, num_inner_episodes, inner_steps = rewards.shape
    flat_rewards = rewards.reshape(batch_size, num_inner_episodes * inner_steps)
    rewards_time_major = flat_rewards.T

    def body(carry, xs):
        discounted_reward, global_discounted = carry
        idx, reward_t = xs
        discounted_reward = jp.where(
            (idx != 0) & (idx % inner_episode_length == 0),
            jp.ones_like(discounted_reward) * global_discounted,
            discounted_reward,
        )
        discounted_reward = reward_t + gamma * discounted_reward
        global_discounted = reward_t.mean() + gamma * global_discounted
        return (discounted_reward, global_discounted), discounted_reward

    (_, _), returns_reversed = jax.lax.scan(
        body,
        (jp.zeros((batch_size,), dtype=rewards.dtype), jp.array(0., dtype=rewards.dtype)),
        (jp.arange(rewards_time_major.shape[0]), jp.flip(rewards_time_major, axis=0)),
    )
    returns = jp.flip(returns_reversed, axis=0).T
    returns = returns.reshape(batch_size, num_inner_episodes, inner_steps)
    return (returns - returns.mean()) / (returns.std() + 1e-5)


@partial(jax.jit, static_argnames=('inner_episode_length', 'player_id'))
def build_policy_data(episodes, gamma, inner_episode_length, player_id):
    batch_size, metric_episode_length = episodes['act'].shape[:2]
    num_inner_episodes = metric_episode_length // inner_episode_length
    state_seq = episodes['mfos_state'][:, :, player_id].reshape(
        batch_size,
        num_inner_episodes,
        inner_episode_length,
        -1,
    )
    action_seq = episodes['act'][:, :, player_id].reshape(batch_size, num_inner_episodes, inner_episode_length)
    taken_logps = jp.take_along_axis(episodes['logp'], episodes['act'][..., None], axis=-1)[..., 0]
    old_logp_seq = taken_logps[:, :, player_id].reshape(batch_size, num_inner_episodes, inner_episode_length)
    rewards = episodes['rew'][:, :, player_id].reshape(batch_size, num_inner_episodes, inner_episode_length)
    returns = mfos_discounted_returns(rewards, gamma, inner_episode_length)
    return {
        'state_seq': state_seq,
        'action_seq': action_seq,
        'old_logp_seq': old_logp_seq,
        'return_seq': returns,
    }


def make_update_mfos_policy_fn(
        optimizer,
        gamma,
        eps_clip,
        entropy_weight,
        num_epochs,
        clip_grad_norm,
        inner_episode_length,
):
    @partial(jax.jit, static_argnames=('player_id',))
    def update(agent, opt_state, episodes, player_id):
        data = build_policy_data(episodes, gamma, inner_episode_length, player_id)

        def loss_fn(a):
            outputs = a.evaluate_agent_sequences({'state_seq': data['state_seq']})
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
            total_loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy
            return total_loss, {
                'loss_mfos_policy': policy_loss,
                'loss_mfos_value': value_loss,
                'mfos_entropy': entropy,
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
                'loss_mfos_total': loss,
                'grad_mfos_norm': grad_norm,
            }
            return (a, state), aux

        (agent, opt_state), metrics = jax.lax.scan(
            epoch_body,
            (agent, opt_state),
            xs=jp.arange(num_epochs),
        )
        return agent, opt_state, jax.tree_map(lambda x: x.mean(), metrics)

    return update


@jax.jit
def compute_mfos_diagnostics(episodes, theta_metrics):
    actions_one_hot = jax.nn.one_hot(episodes['act'], 4).mean(axis=(0, 1))
    return {
        'theta_mean': theta_metrics.mean(),
        'theta_std': theta_metrics.std(),
        'action_frequency_left_player0': actions_one_hot[0, 0],
        'action_frequency_right_player0': actions_one_hot[0, 1],
        'action_frequency_up_player0': actions_one_hot[0, 2],
        'action_frequency_down_player0': actions_one_hot[0, 3],
        'action_frequency_left_player1': actions_one_hot[1, 0],
        'action_frequency_right_player1': actions_one_hot[1, 1],
        'action_frequency_up_player1': actions_one_hot[1, 2],
        'action_frequency_down_player1': actions_one_hot[1, 3],
    }


def compute_mfos_iteration_metric_row(
        episode_batches,
        iteration,
        episode,
        timestep,
        window_start_iteration,
        window_start_episode,
        window_start_timestep,
):
    rewards = np.concatenate([np.asarray(jax.device_get(batch['rew'])) for batch in episode_batches], axis=0)
    same_pickup = np.concatenate(
        [np.asarray(jax.device_get(batch['same_pickup'])) for batch in episode_batches],
        axis=0,
    )
    different_pickup = np.concatenate(
        [np.asarray(jax.device_get(batch['different_pickup'])) for batch in episode_batches],
        axis=0,
    )
    actions = np.concatenate([np.asarray(jax.device_get(batch['act'])) for batch in episode_batches], axis=0)
    num_episodes, game_length, _ = rewards.shape
    pickups = same_pickup + different_pickup
    any_pickup = pickups > 0

    row = {
        'iteration': iteration,
        'episode': episode,
        'timestep': timestep,
        'window_start_iteration': window_start_iteration,
        'window_start_episode': window_start_episode,
        'window_start_timestep': window_start_timestep,
        'num_iterations_in_window': iteration - window_start_iteration + 1,
        'num_episodes_in_window': num_episodes,
        'num_timesteps_in_window': num_episodes * game_length,
        'game_length': game_length,
    }

    for player in (0, 1):
        player_rewards = rewards[:, :, player]
        same = same_pickup[:, :, player].sum(axis=1)
        different = different_pickup[:, :, player].sum(axis=1)
        total_pickups = same + different
        row[f'mean_reward_player{player}'] = float(player_rewards.mean(axis=1).mean())
        row[f'avg_episode_return_player{player}'] = float(player_rewards.sum(axis=1).mean())
        row[f'avg_pickups_player{player}'] = float(total_pickups.mean())
        row[f'avg_same_color_pickups_player{player}'] = float(same.mean())
        row[f'avg_different_color_pickups_player{player}'] = float(different.mean())
        row[f'same_color_pickup_ratio_player{player}'] = float(same.sum() / max(total_pickups.sum(), 1e-8))
        row[f'mean_rewards_{player}'] = row[f'mean_reward_player{player}']
        row[f'mean_pickup_rewards_{player}'] = float(
            (player_rewards * any_pickup[:, :, player]).sum() / max(any_pickup[:, :, player].sum(), 1e-8)
        )
        action_counts = np.bincount(actions[:, :, player].reshape(-1), minlength=4).astype(np.float64)
        marginal = action_counts / max(action_counts.sum(), 1.0)
        row[f'action_entropy_{player}'] = float(-(marginal * np.log(marginal + 1e-8)).sum())
        row[f'easymisses_{player}'] = 0.0
        row[f'adversity_{player}'] = float(different_pickup[:, :, player].sum() / max(pickups[:, :, player].sum(), 1e-8))
        row[f'adversarial_pickup_div_timesteps_{player}'] = float(different_pickup[:, :, player].sum() / (num_episodes * game_length))
        row[f'any_pickup_div_timesteps_{player}'] = float(pickups[:, :, player].sum() / (num_episodes * game_length))
        row[f'adversarial_pickup_div_all_pickup_{player}'] = row[f'adversity_{player}']
        row[f'own_pickup_div_timesteps_{player}'] = float(same_pickup[:, :, player].sum() / (num_episodes * game_length))
        row[f'nearpasses_{player}'] = 0.0

    row['collective_return'] = float(
        row['avg_episode_return_player0'] + row['avg_episode_return_player1']
    ) / 2
    row['reward_difference'] = float(abs(row['avg_episode_return_player0'] - row['avg_episode_return_player1']))
    return row


def set_up_state_from_config(hp):
    dummy_env, _ = CoinGame.init(
        rng=rax.PRNGKey(hp['seed']),
        **coin_game_params(hp),
    )
    mfos_input_shape = (7, hp['game']['height'], hp['game']['width'])
    inner_episode_length = get_mfos_inner_episode_length(hp)
    num_inner_episodes = hp['game']['game_length'] // inner_episode_length
    dummy_state_seq = jp.zeros(
        (
            hp['batch_size'],
            num_inner_episodes,
            inner_episode_length,
            int(np.prod(mfos_input_shape)),
        ),
        dtype=jp.float32,
    )
    rng, agent0_rng, agent1_rng = rax.split(rax.PRNGKey(hp['seed']), 3)

    mfos_hp = hp['mfos']
    agent_module = MFOSCoinAgent(
        input_shape=tuple(mfos_input_shape),
        num_actions=dummy_env.NUM_ACTIONS,
        hidden_size=int(mfos_hp['hidden_size']),
        out_channels=int(mfos_hp['out_channels']),
        layers_before_gru=int(mfos_hp['layers_before_gru']),
    )
    agent0_params = agent_module.init(agent0_rng, {'state_seq': dummy_state_seq, 'rng': rax.PRNGKey(0)})
    agent1_params = agent_module.init(agent1_rng, {'state_seq': dummy_state_seq, 'rng': rax.PRNGKey(1)})
    agent0 = CoinAgent(params=agent0_params, model=agent_module, player=0)
    agent1 = CoinAgent(params=agent1_params, model=agent_module, player=1)
    optimizer = optax.adam(float(mfos_hp['lr']))
    agent0_opt = Optimizer(optimizer, optimizer.init(agent0))
    agent1_opt = Optimizer(optimizer, optimizer.init(agent1))
    update_policy = make_update_mfos_policy_fn(
        optimizer,
        float(hp['reward_discount']),
        float(mfos_hp['eps_clip']),
        float(mfos_hp['entropy_weight']),
        int(mfos_hp['ppo_epochs']),
        float(mfos_hp['clip_grad_norm']) if 'clip_grad_norm' in mfos_hp else None,
        inner_episode_length,
    )
    return {
        'rng': rng,
        'agent0': agent0,
        'agent1': agent1,
        'agent0_opt': agent0_opt,
        'agent1_opt': agent1_opt,
        'update_policy': update_policy,
        'mfos_input_shape': tuple(mfos_input_shape),
    }


def validate_mfos_config(hp):
    if not hp['just_self_play']:
        raise ValueError('mfos_coin_train.py currently implements MFOS self-play only.')
    if hp['agent_0'] != 'mfos' or hp['agent_1'] != 'mfos':
        raise ValueError('Set hp.agent_0=mfos and hp.agent_1=mfos.')
    if hp['game']['height'] != 3 or hp['game']['width'] != 3:
        raise ValueError('This MFOS entry is intended for the same 3x3 Coin Game setting as LOQA self-play.')
    if hp['game']['game_length'] != 200:
        raise ValueError('This MFOS entry expects 200 steps per episode, matching the LOQA self-play setup.')
    inner_episode_length = get_mfos_inner_episode_length(hp)
    if inner_episode_length <= 0:
        raise ValueError('hp.mfos.inner_episode_length must be positive.')
    if hp['game']['game_length'] % inner_episode_length != 0:
        raise ValueError(
            'hp.mfos.inner_episode_length must divide hp.game.game_length so every metric episode has '
            'a fixed 200-step window.'
        )


def train(hp, log_wandb):
    validate_mfos_config(hp)
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
    inner_episode_length = get_mfos_inner_episode_length(hp)
    inner_episodes_per_metric_episode = episode_length // inner_episode_length
    expected_metric_rows = int(np.ceil(total_training_timesteps / metrics_log_timestep_freq))
    state = set_up_state_from_config(hp)

    print('****mfos self play (jax)****')
    print(f'num training iterations: {num_training_iterations}')
    print(f'episodes per iteration: {episodes_per_iteration}')
    print(f'episode length: {episode_length} timesteps')
    print(
        f'mfos inner episode length: {inner_episode_length} timesteps '
        f'({inner_episodes_per_metric_episode} inner episodes per 200-step metric episode)'
    )
    print(
        f'timesteps per iteration/window: {timesteps_per_iteration} '
        f'({episodes_per_iteration} episodes x {episode_length} timesteps)'
    )
    print(f'total training episodes: {total_training_episodes}')
    print(f'total training timesteps: {total_training_timesteps}')
    print(f'metrics log timestep frequency: {metrics_log_timestep_freq}')
    print(f'expected CSV/W&B rows: {expected_metric_rows}')

    iteration_metrics_csv_path = init_mfos_metrics_csv(save_path, hp)
    print(f'metrics csv path: {iteration_metrics_csv_path}')

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
        episodes, theta_metrics = generate_mfos_episodes(
            state['agent0'],
            state['agent1'],
            rng,
            hp,
            state['mfos_input_shape'],
        )
        metric_window_batches.append(episodes)

        new_agent0, new_opt0_state, update0_metrics = state['update_policy'](
            state['agent0'],
            state['agent0_opt'].opt_state,
            episodes,
            0,
        )
        new_agent1, new_opt1_state, update1_metrics = state['update_policy'](
            state['agent1'],
            state['agent1_opt'].opt_state,
            episodes,
            1,
        )
        state['agent0'] = new_agent0
        state['agent1'] = new_agent1
        state['agent0_opt'] = state['agent0_opt'].replace(opt_state=new_opt0_state)
        state['agent1_opt'] = state['agent1_opt'].replace(opt_state=new_opt1_state)
        update_metric_row = {
            key: float(np.mean([scalar_mean(update0_metrics[key]), scalar_mean(update1_metrics[key])]))
            for key in update0_metrics
        }
        diagnostics = compute_mfos_diagnostics(episodes, theta_metrics)
        update_metric_row.update({key: scalar_mean(value) for key, value in diagnostics.items()})

        completed_episodes += episodes_per_iteration
        completed_timesteps += timesteps_per_iteration
        update_metric_row['walltime'] = time.time() - start_time
        metric_window_update_rows.append(update_metric_row)

        should_log_metrics = (
            completed_timesteps - last_metric_timestep >= metrics_log_timestep_freq
            or i == num_training_iterations - 1
        )
        if should_log_metrics:
            metric_row = compute_mfos_iteration_metric_row(
                episode_batches=metric_window_batches,
                iteration=i,
                episode=completed_episodes,
                timestep=completed_timesteps,
                window_start_iteration=window_start_iteration,
                window_start_episode=window_start_episode,
                window_start_timestep=window_start_timestep,
            )
            for key in MFOS_METRIC_FIELDNAMES + ['walltime']:
                values = [row[key] for row in metric_window_update_rows if key in row]
                if values:
                    metric_row[key] = float(np.mean(values))

            append_mfos_metric_row(iteration_metrics_csv_path, metric_row)
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
            print({
                key: round(value, 4)
                for key, value in compute_mfos_iteration_metric_row(
                    [episodes],
                    i,
                    completed_episodes,
                    completed_timesteps,
                    i,
                    completed_episodes - episodes_per_iteration,
                    completed_timesteps - timesteps_per_iteration,
                ).items()
                if key in (
                    'avg_same_color_pickups_player0',
                    'avg_same_color_pickups_player1',
                    'avg_different_color_pickups_player0',
                    'avg_different_color_pickups_player1',
                    'same_color_pickup_ratio_player0',
                    'same_color_pickup_ratio_player1',
                    'collective_return',
                    'reward_difference',
                )
            })

        if i % hp['save_every'] == 0:
            minimal_state = {
                'agent0': npify(state['agent0']),
                'agent1': npify(state['agent1']),
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
        wandb_id = wandb.util.generate_id()
        run_name = get_metrics_csv_filename(hp).removesuffix('.csv')
        wandb.init(
            project='loqa-ipd',
            id=wandb_id,
            name=run_name,
            dir=cfg.wandb.wandb_dir,
            tags=cfg.wandb.tags,
        )
        wandb.config.update(hp)
        wandb.run.log_code('.', include_fn=lambda path: path.endswith('.py'))
        shutil.make_archive('conf', 'zip', 'conf')
        wandb.save('conf.zip', policy='now')
        wandb.run.summary.update(slurm_infos())

    train(hp, log_wandb)


if __name__ == '__main__':
    main()
