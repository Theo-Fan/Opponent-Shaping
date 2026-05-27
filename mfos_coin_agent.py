from flax import linen as nn
import jax
from jax import numpy as jp, random as rax

from ipd import POLAGRU


class MFOSStateEncoder(nn.Module):
    input_shape: tuple
    hidden_size: int
    out_channels: int

    @nn.compact
    def __call__(self, state):
        x = state.reshape(self.input_shape)
        x = jp.transpose(x, (1, 2, 0))
        x = jp.pad(x, ((1, 1), (1, 1), (0, 0)), mode='wrap')
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='VALID')(x)
        x = nn.relu(x)
        x = jp.pad(x, ((1, 1), (1, 1), (0, 0)), mode='wrap')
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='VALID')(x)
        x = nn.relu(x)
        x = x.reshape(-1)
        x = nn.Dense(self.hidden_size)(x)
        return nn.relu(x)


class MFOSCoinAgent(nn.Module):
    input_shape: tuple
    num_actions: int
    hidden_size: int
    out_channels: int
    layers_before_gru: int

    def setup(self):
        self.actor_encoder = MFOSStateEncoder(self.input_shape, self.hidden_size, self.out_channels)
        self.value_encoder = MFOSStateEncoder(self.input_shape, self.hidden_size, self.out_channels)
        self.theta_encoder = MFOSStateEncoder(self.input_shape, self.hidden_size, self.out_channels)
        self.actor_head = POLAGRU(self.hidden_size, self.hidden_size, self.layers_before_gru)
        self.value_head = POLAGRU(self.hidden_size, self.hidden_size, self.layers_before_gru)
        self.theta_head = POLAGRU(self.hidden_size, self.hidden_size, self.layers_before_gru)
        self.actor_logits = nn.Dense(self.num_actions)
        self.value_out = nn.Dense(1)
        self.theta_out = nn.Dense(self.hidden_size)

    def __call__(self, x):
        state_seq = x['state_seq']
        rng = x['rng']
        if state_seq.ndim == 5:
            self.evaluate_batched_meta_sequences({'state_seq': state_seq})
        elif state_seq.ndim == 4:
            self.evaluate_agent_sequences({'state_seq': state_seq})
        else:
            self.evaluate_meta_sequences({'state_seq': state_seq})
        self.call_step({
            'state': state_seq.reshape(-1, state_seq.shape[-1])[0],
            'theta': self.get_initial_theta(),
            'carry_actor': self.actor_head.get_initial_carry(),
            'carry_value': self.value_head.get_initial_carry(),
            'rng': rng,
        })

    def get_initial_carries(self):
        return {
            'carry_actor': self.actor_head.get_initial_carry(),
            'carry_value': self.value_head.get_initial_carry(),
        }

    def get_initial_theta(self):
        return jp.ones((self.hidden_size,), dtype=jp.float32)

    def theta_from_seq(self, x):
        state_seq = x['state_seq']
        features = jax.vmap(self.theta_encoder)(state_seq)
        hs = self.theta_head(features, carry=None)['hs']
        return nn.sigmoid(self.theta_out(hs[-1]))

    def theta_from_batch_seq(self, x):
        state_seq = x['state_seq']
        batch_size, inner_steps, state_dim = state_seq.shape
        flat_state = state_seq.reshape(batch_size * inner_steps, state_dim)
        features = jax.vmap(self.theta_encoder)(flat_state)
        features = features.reshape(batch_size, inner_steps, self.hidden_size)
        last_h = jax.vmap(lambda seq: self.theta_head(seq, carry=None)['hs'][-1])(features)
        return nn.sigmoid(jax.vmap(self.theta_out)(last_h))

    def call_step(self, x):
        state = x['state']
        theta = x['theta']
        actor_features = self.actor_encoder(state)
        actor_res = self.actor_head(actor_features[None, :], carry=x['carry_actor'])
        actor_hidden = nn.relu(actor_res['hs'][0]) * theta
        logp = nn.log_softmax(self.actor_logits(actor_hidden), axis=-1)
        action = rax.categorical(x['rng'], logp)

        value_features = self.value_encoder(state)
        value_res = self.value_head(value_features[None, :], carry=x['carry_value'])
        value_hidden = nn.relu(value_res['hs'][0]) * jax.lax.stop_gradient(theta)
        value = self.value_out(value_hidden)[0]
        return {
            'logp': logp,
            'value': value,
            'action': action,
            'carry_actor': actor_res['carry'],
            'carry_value': value_res['carry'],
        }

    def evaluate_meta_sequences(self, x):
        state_seq = x['state_seq']
        num_players, outer_steps, inner_steps, state_dim = state_seq.shape

        def theta_for_one_outer(seq):
            return self.theta_from_seq({'state_seq': seq})

        theta_current = jax.vmap(jax.vmap(theta_for_one_outer, in_axes=0), in_axes=0)(state_seq)
        theta0 = jp.ones((num_players, 1, self.hidden_size), dtype=state_seq.dtype)
        theta = jp.concatenate([theta0, theta_current[:, :-1]], axis=1)

        flat_state = state_seq.reshape(num_players * outer_steps * inner_steps, state_dim)
        actor_features = jax.vmap(self.actor_encoder)(flat_state)
        value_features = jax.vmap(self.value_encoder)(flat_state)
        actor_features = actor_features.reshape(num_players, outer_steps, inner_steps, self.hidden_size)
        value_features = value_features.reshape(num_players, outer_steps, inner_steps, self.hidden_size)

        def run_actor(seq):
            return self.actor_head(seq, carry=None)['hs']

        def run_value(seq):
            return self.value_head(seq, carry=None)['hs']

        actor_hidden = jax.vmap(jax.vmap(run_actor, in_axes=0), in_axes=0)(actor_features)
        value_hidden = jax.vmap(jax.vmap(run_value, in_axes=0), in_axes=0)(value_features)

        actor_hidden = nn.relu(actor_hidden) * theta[:, :, None, :]
        value_hidden = nn.relu(value_hidden) * jax.lax.stop_gradient(theta[:, :, None, :])

        flat_actor_hidden = actor_hidden.reshape(num_players * outer_steps * inner_steps, self.hidden_size)
        flat_value_hidden = value_hidden.reshape(num_players * outer_steps * inner_steps, self.hidden_size)
        logits = jax.vmap(self.actor_logits)(flat_actor_hidden)
        values = jax.vmap(self.value_out)(flat_value_hidden)[..., 0]

        logp_seq = nn.log_softmax(logits, axis=-1).reshape(num_players, outer_steps, inner_steps, self.num_actions)
        value_seq = values.reshape(num_players, outer_steps, inner_steps)
        return {'logp_seq': logp_seq, 'value_seq': value_seq, 'theta_seq': theta}

    def evaluate_batched_meta_sequences(self, x):
        state_seq = x['state_seq']
        num_players, batch_size, outer_steps, inner_steps, state_dim = state_seq.shape

        def theta_for_one_outer(seq):
            return self.theta_from_seq({'state_seq': seq})

        theta_current = jax.vmap(
            jax.vmap(
                jax.vmap(theta_for_one_outer, in_axes=0),
                in_axes=0,
            ),
            in_axes=0,
        )(state_seq)
        theta0 = jp.ones((num_players, batch_size, 1, self.hidden_size), dtype=state_seq.dtype)
        theta = jp.concatenate([theta0, theta_current[:, :, :-1]], axis=2)

        flat_state = state_seq.reshape(num_players * batch_size * outer_steps * inner_steps, state_dim)
        actor_features = jax.vmap(self.actor_encoder)(flat_state)
        value_features = jax.vmap(self.value_encoder)(flat_state)
        actor_features = actor_features.reshape(
            num_players, batch_size, outer_steps, inner_steps, self.hidden_size
        )
        value_features = value_features.reshape(
            num_players, batch_size, outer_steps, inner_steps, self.hidden_size
        )

        def run_actor(seq):
            return self.actor_head(seq, carry=None)['hs']

        def run_value(seq):
            return self.value_head(seq, carry=None)['hs']

        actor_hidden = jax.vmap(
            jax.vmap(jax.vmap(run_actor, in_axes=0), in_axes=0),
            in_axes=0,
        )(actor_features)
        value_hidden = jax.vmap(
            jax.vmap(jax.vmap(run_value, in_axes=0), in_axes=0),
            in_axes=0,
        )(value_features)

        actor_hidden = nn.relu(actor_hidden) * theta[:, :, :, None, :]
        value_hidden = nn.relu(value_hidden) * jax.lax.stop_gradient(theta[:, :, :, None, :])

        flat_actor_hidden = actor_hidden.reshape(
            num_players * batch_size * outer_steps * inner_steps, self.hidden_size
        )
        flat_value_hidden = value_hidden.reshape(
            num_players * batch_size * outer_steps * inner_steps, self.hidden_size
        )
        logits = jax.vmap(self.actor_logits)(flat_actor_hidden)
        values = jax.vmap(self.value_out)(flat_value_hidden)[..., 0]

        logp_seq = nn.log_softmax(logits, axis=-1).reshape(
            num_players, batch_size, outer_steps, inner_steps, self.num_actions
        )
        value_seq = values.reshape(num_players, batch_size, outer_steps, inner_steps)
        return {'logp_seq': logp_seq, 'value_seq': value_seq, 'theta_seq': theta}

    def evaluate_agent_sequences(self, x):
        state_seq = x['state_seq']
        batch_size, outer_steps, inner_steps, state_dim = state_seq.shape

        theta_current = jax.vmap(
            lambda seq: self.theta_from_batch_seq({'state_seq': seq}),
            in_axes=1,
            out_axes=1,
        )(state_seq)
        theta0 = jp.ones((batch_size, 1, self.hidden_size), dtype=state_seq.dtype)
        theta = jp.concatenate([theta0, theta_current[:, :-1]], axis=1)

        flat_state = state_seq.reshape(batch_size * outer_steps * inner_steps, state_dim)
        actor_features = jax.vmap(self.actor_encoder)(flat_state)
        value_features = jax.vmap(self.value_encoder)(flat_state)
        actor_features = actor_features.reshape(batch_size, outer_steps, inner_steps, self.hidden_size)
        value_features = value_features.reshape(batch_size, outer_steps, inner_steps, self.hidden_size)

        def run_actor(seq):
            return self.actor_head(seq, carry=None)['hs']

        def run_value(seq):
            return self.value_head(seq, carry=None)['hs']

        actor_hidden = jax.vmap(jax.vmap(run_actor, in_axes=0), in_axes=0)(actor_features)
        value_hidden = jax.vmap(jax.vmap(run_value, in_axes=0), in_axes=0)(value_features)

        actor_hidden = nn.relu(actor_hidden) * theta[:, :, None, :]
        value_hidden = nn.relu(value_hidden) * jax.lax.stop_gradient(theta[:, :, None, :])

        flat_actor_hidden = actor_hidden.reshape(batch_size * outer_steps * inner_steps, self.hidden_size)
        flat_value_hidden = value_hidden.reshape(batch_size * outer_steps * inner_steps, self.hidden_size)
        logits = jax.vmap(self.actor_logits)(flat_actor_hidden)
        values = jax.vmap(self.value_out)(flat_value_hidden)[..., 0]

        logp_seq = nn.log_softmax(logits, axis=-1).reshape(
            batch_size, outer_steps, inner_steps, self.num_actions
        )
        value_seq = values.reshape(batch_size, outer_steps, inner_steps)
        return {'logp_seq': logp_seq, 'value_seq': value_seq, 'theta_seq': theta}
