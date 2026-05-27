# LOQA: Learning With Opponent Q-Learning Awareness
This is the repository that contains the codebase of LOQA (Learning with Opponent Q-Learning Awareness) paper.
It may be surprising a little, but Naïve RL fails badly on general sum games. One subset that shows this clearly are social dilemmas like IPD and Coin Game. Most of the games in the real world are actually general sum and many of the most important ones, countries negotiating, bussiness deals, scientists deciding what to disclose and what not to, etc is general sum and also a social dilemma. Methods like LOLA and POLA shed light on the reason behind this failure but their methods require constructing explicit huge optimization graphs of the opponent's optimization and differentiating through it and therefore not scalable. 
LOQA sidesteps modeling explicitly the opponent's optimization by realising that the op's optimization is following the opponent's return. Therefore, by shaping the return (Q-values) we shape the opponent.
![Multi Agent RL is significantly behind in solving General Sum Games](assets/loqa-meme.jpeg)
LOQA is the most scalable Opponent Shaping Algorithm yet improving the speed of finding reciprocation-based cooperative policies by two orders of magnitude compared to previous SOTA which is POLA. Also, on larger games where POLA fails, LOQA succeeds. In a world where agents will be used to help us in decision making, making sure their optimization can actually solve the real world's general sum games is very important. Therefore:
![The world is saved by LOQA](assets/world-without-loqa.jpeg)
## Installing the environment

LOQA's code has been written with `jax` and `flax` as the neural network library.
If you just want to run the code on cpu, for debugging or playing around purposes, then you most probably will only
need to run `pip install -r requirements.txt` and then follow the provided examples. 

If you want to run the code on gpu, then you will need to install the
appropriate version of jax and flax for your specific gpu. Usually, it is a bit of pain. Especially, if you're on a computing cluster where you don't have sudo access. Refer to installation instructions from the [jax](
https://github.com/google/jax) and [flax](https://github.com/google/flax) repositories. I hope google team makes the installation of jax as easy as pytorch one day. 

Also, while the code most probably will be fine with newer versions of `jax` and `flax` there is a slight chance it breaks because these two are not backward compatible. In that case, just install the older one, or you may need to do some code changes (which understandably is not on anyone's wishlist. But, it happens sometimes. During the development of this own project, the flax UI for GRUs changed. First, I thought I somehow have corrupted the code and took me a long time to understand it was just a simple change).

# Understanding the LOQA's Implementation
If you want to implement LOQA as a basline or if you want to improve upon LOQA, we recommend studying the `ipd.py` first. This a self-containing script that runs LOQA on IPD. After that understanding `coin_train.py` should be easier. Also, you can jump straight into the coin script. The main LOQA magic is happening in the `agent_policy_loss` method. LOQA can have different implementations, based on how you make the rewards differentiable. `n-step` just makes the next n rewards differentiable. However, `loaded-dice` is the main condition used in LOQA results that makes the whole return differentiable but it is harder to understan. In order to understand what is happening with the DICE, you can study the DICE and Loaded-DICE paper. However, just keep in mind at the end DICE is an engineering trick. All of these terms coudl have been written as expectations like how we write REINFORCE. But DICE makes this much easier. Just note that the `magic_box` gives you 1 in the forward pass, but in the backward pass it switches to log probability. That is the key to understanding DICE. If you have any questions, just message me on my email: mi[dot]aghajohari[at]gmail[dot]com


# Recreating the LOQA Paper Results
This repository is configured to run LOQA self-play on the modified 3x3 Coin Game:

- each episode has 200 environment steps;
- each CSV/W&B row averages 20 episodes, i.e. 4000 environment timesteps;
- total training lasts 3e7 environment timesteps;
- metrics are written to `experiments/<run_id>/selfplay_LOQA_seed<seed>.csv`;
- when W&B is enabled, `<run_id>` is the W&B run id;
- W&B metrics are grouped with slash prefixes, for example `agent0/same_color_pickups`, `agent1/different_color_pickups`, `agent0/same_color_pickup_ratio`, `agent1/mean_reward`, `global/reward_difference`, and `global/average_reward`.

run:
```bash
conda activate socialjax 
cd LOQA/
python coin_train.py hp=loqa_iclr wandb.state=enabled "wandb.tags=[coin,loqa_3x3]"
```

For a short smoke test without W&B:
```sh
python coin_train.py hp=loqa_iclr wandb.state=disabled hp.max_train_timestep=4000 hp.save_every=999999 hp.eval_every=999999
```

---



If experiment for 5 seeds, run:
```bash
cd LOQA/
for seed in 21 22 23 24 25; do
  python coin_train.py hp=loqa_iclr hp.seed=$seed wandb.state=enabled "wandb.tags=[coin,loqa_3x3,seed${seed}]"
done
```

Collecting exp results, run: 

```sh
cd LOQA/

for seed in 21 22 23 24 25; do
  src=$(find experiments -name "selfplay_LOQA_seed${seed}.csv" | head -n 1)
  if [ -n "$src" ]; then
    cp "$src" ~/
    echo "copied seed ${seed}: $src -> ~/selfplay_LOQA_seed${seed}.csv"
  else
    echo "missing seed ${seed}"
  fi
done
```

## JAX Reciprocator Self-Play on the Same Coin Game

The JAX Reciprocator entry point uses the same modified 3x3 Coin Game setup as the LOQA command above: 200 steps per episode, 20 episodes per metric row, 4000 timesteps per CSV/W&B write, and 3e7 total environment timesteps. The default policy is a convolutional encoder plus GRU, closer to the original Reciprocator coin-game actor-critic than the earlier flatten-observation GRU.

Run one seed:
```bash
conda activate socialjax
cd LOQA/
python reciprocator_coin_train.py hp=reciprocator wandb.state=enabled "wandb.tags=[coin,reciprocator_3x3]"
```

Run a short smoke test without W&B:
```bash
python reciprocator_coin_train.py hp=reciprocator wandb.state=disabled hp.max_train_timestep=4000 hp.reciprocator.influence.num_initialization_iterations=0 hp.reciprocator.influence.target_period=1 hp.reciprocator.influence.target_epochs=1 hp.reciprocator.influence.num_train_batches=1 hp.reciprocator.influence.target_batch_size=512 hp.save_every=999999 hp.eval_every=999999
```

Run five seeds:
```bash
cd LOQA/
for seed in 21 22 23 24 25; do
  python reciprocator_coin_train.py hp=reciprocator hp.seed=$seed wandb.state=enabled "wandb.tags=[coin,reciprocator_3x3,seed${seed}]"
done
```

Reciprocator CSVs are saved as `experiments/<run_id>/selfplay_RECIPROCATOR_seed<seed>.csv`.
The CSV also logs Reciprocator diagnostics such as reciprocal-reward std/mean-absolute value, shaped returns, pickup correlations, and per-action frequencies.

## JAX MFOS Self-Play on the Same Coin Game

The JAX MFOS entry point uses the same 3x3 Coin Game, 200-step metric episodes, 20 metric episodes per CSV/W&B row, 4000 timesteps per row, and 3e7 total environment timesteps. The default run writes 7500 CSV/W&B rows. Internally it follows the official MFOS self-play structure with two independent MFOS learners, separate optimizers, environment resets at each MFOS inner-episode boundary, batch-averaged theta updates from the previous inner trajectory, and MFOS-style PPO returns. Each 200-step metric episode is aggregated from 10 MFOS inner episodes of 20 steps (`hp.mfos.inner_episode_length=20`), so metrics remain computed over fixed 200-step windows.

For smoother MFOS curves, CSV/W&B logs every 4000 timesteps and PPO also updates every logged metric window by default (`hp.mfos.update_interval_iterations=1`). The default MFOS PPO step is less aggressive than the earlier unstable setting but still learns quickly: `lr=5e-4`, linearly decayed to `lr_final=1e-4`, `ppo_epochs=8`, `eps_clip=0.15`, and `clip_grad_norm=0.5`.

Run one seed:
```bash
conda activate socialjax
cd LOQA/
python mfos_coin_train.py hp=mfos wandb.state=enabled "wandb.tags=[coin,mfos_3x3]"
```

Run one seed with explicit stability overrides:
```bash
python mfos_coin_train.py hp=mfos hp.mfos.update_interval_iterations=1 hp.mfos.lr=0.0005 hp.mfos.lr_final=0.0001 hp.mfos.ppo_epochs=8 hp.mfos.eps_clip=0.15 hp.mfos.clip_grad_norm=0.5 wandb.state=enabled "wandb.tags=[coin,mfos_3x3,stable]"
```

Run a short smoke test without W&B:
```bash
python mfos_coin_train.py hp=mfos wandb.state=disabled hp.max_train_timestep=4000 hp.save_every=999999 hp.eval_every=999999
```

Run five seeds:
```bash
cd LOQA/
for seed in 21 22 23 24 25; do
  python mfos_coin_train.py hp=mfos hp.seed=$seed wandb.state=enabled "wandb.tags=[coin,mfos_3x3,seed${seed}]"
done
```

MFOS CSVs are saved as `experiments/<run_id>/selfplay_MFOS_seed<seed>.csv`.
The CSV also logs MFOS diagnostics such as PPO loss, entropy, gradient norm, theta mean/std, and per-action frequencies.





