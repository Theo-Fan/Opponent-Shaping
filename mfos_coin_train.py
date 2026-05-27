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
from coin_game import CoinGame, coin_game_params, episode_stats, make_zero_episode
from coin_train import (
    Optimizer,
    TRAIN_ITERATION_METRICS_FIELDNAMES,
    compute_iteration_metric_row,
    get_metrics_csv_filename,
    get_metrics_log_timestep_freq,
    get_num_training_episodes_per_iteration,
    get_num_training_iterations,
    get_num_training_timesteps_per_iteration,
    get_total_training_episodes,
    get_total_training_timesteps,
    scalar_mean,
    tree_concatenate,
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


def discounted_returns_flat(rewards, gamma):
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


def make_zero_mfos_episode(trace_length, coin_game, mfos_input_shape):
    episode = make_zero_episode(trace_length=trace_length, coin_game=coin_game)
    episode['mfos_state'] = jp.zeros(
        [trace_length, 2, *mfos_input_shape],
        dtype=episode['obs'].dtype,
    )
    return episode


def get_mfos_inner_episode_length(hp):
    return int(hp['mfos'].get('inner_episode_length', hp['game']['game_length']))


@partial(
    jax.jit,
    static_argnames=(
        'metric_episode_length',
        'inner_episode_length',
        'height',
        'width',
        'mfos_input_shape',
    ),
)
def play_mfos_metric_episode(
        agent,
        rng,
        env,
        metric_episode_length,
        inner_episode_length,
        height,
        width,
        mfos_input_shape,
):
    episode = make_zero_mfos_episode(metric_episode_length, env, mfos_input_shape)
    episode['obs'] = episode['obs'].at[0].set(env.get_obs())
    episode['coin_pos'] = episode['coin_pos'].at[0].set(env.coin_pos)
    episode['coin_owner'] = episode['coin_owner'].at[0].set(env.coin_owner)
    episode['player1_pos'] = episode['player1_pos'].at[0].set(env.players_pos[0])
    episode['player2_pos'] = episode['player2_pos'].at[0].set(env.players_pos[1])
    previous_rewards = jp.zeros((2,), dtype=episode['rew'].dtype)
    theta_init = agent.get_initial_theta()

    def scan_inner_episode(carry, _):
        env, rng, episode, global_t, previous_rewards, theta0, theta1 = carry
        carries0 = agent.get_initial_carries()
        carries1 = agent.get_initial_carries()

        def body_fn(inner_carry, _):
            env, rng, episode, t, previous_rewards, c0_actor, c0_value, c1_actor, c1_value = inner_carry
            rng, rng0, rng1 = rax.split(rng, 3)
            episode['games'] = jax.tree_map(lambda x, o: x.at[t].set(o), episode['games'], env)

            state0 = make_mfos_state(
                episode['obs'][t, 0],
                previous_rewards[0],
                previous_rewards[1],
                0.,
                height,
                width,
            )
            state1 = make_mfos_state(
                episode['obs'][t, 1],
                previous_rewards[1],
                previous_rewards[0],
                0.,
                height,
                width,
            )
            out0 = agent.call_step({
                'state': state0.reshape(-1),
                'theta': theta0,
                'carry_actor': c0_actor,
                'carry_value': c0_value,
                'rng': rng0,
            })
            out1 = agent.call_step({
                'state': state1.reshape(-1),
                'theta': theta1,
                'carry_actor': c1_actor,
                'carry_value': c1_value,
                'rng': rng1,
            })
            actions = jp.stack([out0['action'], out1['action']])
            logp = jp.stack([out0['logp'], out1['logp']], axis=0)

            env, obs, rewards = env.step(actions)
            rewards = rewards.astype(previous_rewards.dtype)
            episode['obs'] = episode['obs'].at[t + 1].set(obs)
            episode['coin_pos'] = episode['coin_pos'].at[t + 1].set(env.coin_pos)
            episode['coin_owner'] = episode['coin_owner'].at[t + 1].set(env.coin_owner)
            episode['player1_pos'] = episode['player1_pos'].at[t + 1].set(env.players_pos[0])
            episode['player2_pos'] = episode['player2_pos'].at[t + 1].set(env.players_pos[1])
            episode['mfos_state'] = episode['mfos_state'].at[t, 0].set(state0)
            episode['mfos_state'] = episode['mfos_state'].at[t, 1].set(state1)
            episode['act'] = episode['act'].at[t].set(actions)
            episode['logp'] = episode['logp'].at[t].set(logp)
            episode['rew'] = episode['rew'].at[t].set(rewards)
            states = jp.stack([state0, state1], axis=0)
            return (
                env,
                rng,
                episode,
                t + 1,
                rewards,
                out0['carry_actor'],
                out0['carry_value'],
                out1['carry_actor'],
                out1['carry_value'],
            ), states

        init = (
            env,
            rng,
            episode,
            global_t,
            previous_rewards,
            carries0['carry_actor'],
            carries0['carry_value'],
            carries1['carry_actor'],
            carries1['carry_value'],
        )
        (env, rng, episode, global_t, previous_rewards, _, _, _, _), chunk_states = jax.lax.scan(
            body_fn,
            init,
            xs=(),
            length=inner_episode_length,
        )
        theta0_next = agent.theta_from_seq({
            'state_seq': chunk_states[:, 0].reshape(inner_episode_length, -1),
        })
        theta1_next = agent.theta_from_seq({
            'state_seq': chunk_states[:, 1].reshape(inner_episode_length, -1),
        })
        return (
            env,
            rng,
            episode,
            global_t,
            previous_rewards,
            theta0_next,
            theta1_next,
        ), jp.stack([theta0_next, theta1_next])

    init = (
        env,
        rng,
        episode,
        0,
        previous_rewards,
        theta_init,
        theta_init,
    )
    num_inner_episodes = metric_episode_length // inner_episode_length
    (env, _, episode, _, _, _, _), theta_metrics = jax.lax.scan(
        scan_inner_episode,
        init,
        xs=(),
        length=num_inner_episodes,
    )
    episode['games'] = jax.tree_map(lambda x, o: x.at[metric_episode_length].set(o), episode['games'], env)
    return episode, theta_metrics


@partial(jax.jit, static_argnames=('hp', 'mfos_input_shape'))
def generate_mfos_episodes(agent, rng, hp, mfos_input_shape):
    batch_size = hp['batch_size']
    rngs = rax.split(rscope(rng, 'mfos_metric_batch'), batch_size)
    metric_episode_length = hp['game']['game_length']
    inner_episode_length = get_mfos_inner_episode_length(hp)

    def generate_one(metric_rng):
        game_rng, play_rng = rax.split(metric_rng)
        env, _ = CoinGame.init(
            rng=game_rng,
            **coin_game_params(hp),
        )
        return play_mfos_metric_episode(
            agent,
            play_rng,
            env,
            metric_episode_length,
            inner_episode_length,
            hp['game']['height'],
            hp['game']['width'],
            mfos_input_shape,
        )

    episodes, theta_metrics = jax.vmap(generate_one)(rngs)
    return episodes, theta_metrics


@partial(jax.jit, static_argnames=('inner_episode_length',))
def build_policy_data(episodes, gamma, inner_episode_length):
    batch_size, metric_episode_length = episodes['act'].shape[:2]
    num_inner_episodes = metric_episode_length // inner_episode_length
    state_seq = episodes['mfos_state'].reshape(
        batch_size,
        num_inner_episodes,
        inner_episode_length,
        2,
        -1,
    )
    state_seq = jp.transpose(state_seq, (3, 0, 1, 2, 4))
    action_seq = episodes['act'].reshape(batch_size, num_inner_episodes, inner_episode_length, 2)
    action_seq = jp.transpose(action_seq, (3, 0, 1, 2))
    taken_logps = jp.take_along_axis(episodes['logp'], episodes['act'][..., None], axis=-1)[..., 0]
    old_logp_seq = taken_logps.reshape(batch_size, num_inner_episodes, inner_episode_length, 2)
    old_logp_seq = jp.transpose(old_logp_seq, (3, 0, 1, 2))
    rewards = episodes['rew'].reshape(batch_size, num_inner_episodes, inner_episode_length, 2)
    rewards = jp.transpose(rewards, (3, 0, 1, 2))
    returns = discounted_returns_flat(rewards.reshape(2 * batch_size, metric_episode_length), gamma)
    returns = returns.reshape(2, batch_size, num_inner_episodes, inner_episode_length)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
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
    @jax.jit
    def update(agent, opt_state, episodes):
        data = build_policy_data(episodes, gamma, inner_episode_length)

        def loss_fn(a):
            outputs = a.evaluate_batched_meta_sequences({'state_seq': data['state_seq']})
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
            2,
            hp['batch_size'],
            num_inner_episodes,
            inner_episode_length,
            int(np.prod(mfos_input_shape)),
        ),
        dtype=jp.float32,
    )
    rng, agent_rng = rax.split(rax.PRNGKey(hp['seed']), 2)

    mfos_hp = hp['mfos']
    agent_module = MFOSCoinAgent(
        input_shape=tuple(mfos_input_shape),
        num_actions=dummy_env.NUM_ACTIONS,
        hidden_size=int(mfos_hp['hidden_size']),
        out_channels=int(mfos_hp['out_channels']),
        layers_before_gru=int(mfos_hp['layers_before_gru']),
    )
    agent_params = agent_module.init(agent_rng, {'state_seq': dummy_state_seq, 'rng': rax.PRNGKey(0)})
    agent = CoinAgent(params=agent_params, model=agent_module, player=0)
    optimizer = optax.adam(float(mfos_hp['lr']))
    agent_opt = Optimizer(optimizer, optimizer.init(agent))
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
        'agent': agent,
        'agent_opt': agent_opt,
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

    dummy_env, _ = CoinGame.init(
        rng=rax.PRNGKey(hp['seed']),
        **coin_game_params(hp),
    )
    episode_stats_jitted = jax.jit(lambda es: episode_stats(es, dummy_env))

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
            state['agent'],
            rng,
            hp,
            state['mfos_input_shape'],
        )
        metric_window_batches.append(episodes)

        new_agent, new_opt_state, update_metrics = state['update_policy'](
            state['agent'],
            state['agent_opt'].opt_state,
            episodes,
        )
        state['agent'] = new_agent
        state['agent_opt'] = state['agent_opt'].replace(opt_state=new_opt_state)
        update_metric_row = {key: scalar_mean(value) for key, value in update_metrics.items()}
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
            for key in MFOS_METRIC_FIELDNAMES + ['walltime']:
                values = [row[key] for row in metric_window_update_rows if key in row]
                if values:
                    metric_row[key] = float(np.mean(values))

            append_mfos_metric_row(iteration_metrics_csv_path, metric_row)
            if log_wandb:
                wandb.log(metric_row, step=completed_timesteps)

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
                'agent': npify(state['agent']),
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
        for root, dirs, files in os.walk('conf'):
            for file in files:
                if file.endswith('.yaml'):
                    wandb.save(os.path.join(root, file))
        shutil.make_archive('conf', 'zip', 'conf')
        wandb.save('conf.zip')
        wandb.run.summary.update(slurm_infos())

    train(hp, log_wandb)


if __name__ == '__main__':
    main()
