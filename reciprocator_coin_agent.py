from flax import linen as nn
from jax import numpy as jp, random as rax

from ipd import POLAGRU


class GRUActorCriticCoinAgent(nn.Module):
    num_actions: int
    hidden_size_actor: int
    hidden_size_value: int
    layers_before_gru_actor: int
    layers_before_gru_value: int

    def setup(self):
        self.actor_head = POLAGRU(self.num_actions, self.hidden_size_actor, self.layers_before_gru_actor)
        self.value_head = POLAGRU(1, self.hidden_size_value, self.layers_before_gru_value)

    def __call__(self, x):
        obs_seq = x['obs_seq']
        rng = x['rng']
        t = x['t']
        self.call_seq({'obs_seq': obs_seq})
        self.call_step({
            'carry_actor': self.actor_head.get_initial_carry(),
            'carry_qvalue': self.value_head.get_initial_carry(),
            'obs': obs_seq[t],
            'rng': rng,
            't': t,
        })

    def get_initial_carries(self):
        return {
            'carry_actor': self.actor_head.get_initial_carry(),
            'carry_qvalue': self.value_head.get_initial_carry(),
        }

    def call_seq(self, x):
        logp_seq = self.logp_seq(x)['logp_seq']
        value_seq = self.value_seq(x)['value_seq']
        return {'logp_seq': logp_seq, 'value_seq': value_seq}

    def logp_seq(self, x):
        obs_seq = x['obs_seq']
        logits_seq = self.actor_head(obs_seq, carry=None)['hs']
        logp_seq = nn.log_softmax(logits_seq, axis=-1)
        return {'logp_seq': logp_seq}

    def value_seq(self, x):
        obs_seq = x['obs_seq']
        t_seq = jp.arange(obs_seq.shape[0], dtype=obs_seq.dtype).reshape(-1, 1)
        value_inputs = jp.concatenate([obs_seq, t_seq], axis=-1)
        value_seq = self.value_head(value_inputs, carry=None)['hs'][..., 0]
        return {'value_seq': value_seq}

    def call_step(self, x):
        obs = x['obs']
        rng = x['rng']
        t = x['t']
        carry_actor = x['carry_actor']
        carry_value = x['carry_qvalue']

        out_actor = self.logp_step({'obs': obs, 'carry': carry_actor})
        logp = out_actor['logp']
        next_carry_actor = out_actor['carry']
        action = rax.categorical(rng, logp)

        out_value = self.value_step({'obs': obs, 't': t, 'carry': carry_value})
        next_carry_value = out_value['carry']

        return {
            'logp': logp,
            'qvalue': jp.repeat(out_value['value'], self.num_actions),
            'carry_actor': next_carry_actor,
            'carry_qvalue': next_carry_value,
            'action': action,
        }

    def logp_step(self, x):
        actor_res = self.actor_head(x=x['obs'][None, :], carry=x['carry'])
        logits = actor_res['hs'][0]
        logp = nn.log_softmax(logits, axis=-1)
        return {'logp': logp, 'carry': actor_res['carry']}

    def value_step(self, x):
        value_input = jp.concatenate([x['obs'], jp.array([x['t']], dtype=x['obs'].dtype)], axis=-1)
        value_res = self.value_head(x=value_input[None, :], carry=x['carry'])
        value = value_res['hs'][0, 0]
        return {'value': value, 'carry': value_res['carry']}


class ScalarPredictor(nn.Module):
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        return nn.Dense(self.output_size)(x)
