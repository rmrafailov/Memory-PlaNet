# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools
from planet.models import base
from dnc.dnc import dnc


FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 100000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval.")




class RSSM(base.Base):
  """Deterministic and stochastic state model.

  The stochastic latent is computed from the hidden state at the same time
  step. If an observation is present, the posterior latent is compute from both
  the hidden state and the observation.

  Prior:    Posterior:

  (a)       (a)
     \         \
      v         v
  [h]->[h]  [h]->[h]
      ^ |       ^ :
     /  v      /  v
  (s)  (s)  (s)  (s)
                  ^
                  :
                 (o)
  """

  def __init__(
      self, state_size, belief_size, embed_size,
      future_rnn=False, mean_only=False, min_stddev=1e-5):
    self._state_size = state_size
    self._belief_size = belief_size
    self._embed_size = embed_size
    self._future_rnn = future_rnn
    #self._cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)

    access_config = {
            "memory_size": 32,
            "word_size": 64,
            "num_reads": 4,
            "num_writes": 4,
            }
    
    controller_config = {
            "hidden_size": self._belief_size,
            }
        
    self._cell = dnc.DNC(access_config, controller_config, self._belief_size)
    self._kwargs = dict(units=self._embed_size, activation=tf.nn.relu)
    self._mean_only = mean_only
    self._min_stddev = min_stddev
    super(RSSM, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))

  @property
  def state_size(self):
    return {
        'mean': self._state_size,
        'stddev': self._state_size,
        'sample': self._state_size,
        'belief': self._belief_size,
        'rnn_state': self._belief_size,
    }

  def dist_from_state(self, state, mask=None):
    """Extract the latent distribution from a prior or posterior state."""
    if mask is not None:
      stddev = tools.mask(state['stddev'], mask, value=1)
    else:
      stddev = state['stddev']
    dist = tfd.MultivariateNormalDiag(state['mean'], stddev)
    return dist

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return tf.concat([state['sample'], state['belief']], -1)

  def divergence_from_states(self, lhs, rhs, mask):
    """Compute the divergence measure between two states."""
    lhs = self.dist_from_state(lhs, mask)
    rhs = self.dist_from_state(rhs, mask)
    return tools.mask(tfd.kl_divergence(lhs, rhs), mask)

  def _transition(self, prev_state, prev_action, zero_obs):
    """Compute prior next state by applying the transition dynamics."""
    print("************************")
    print("************************")
    print("************************")
    print(prev_state)
    print("************************")
    print(prev_action)
    print("************************")
    print("************************")
    print("************************")
    
    
    inputs = tf.concat([prev_state['sample'], prev_action], -1)
    hidden = tf.layers.dense(inputs, **self._kwargs)
    belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
    if self._future_rnn:
      hidden = belief
    hidden = tf.layers.dense(hidden, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
    return {
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
        'belief': belief,
        'rnn_state': rnn_state,
    }

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""
    prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
    inputs = tf.concat([prior['belief'], obs], -1)
    hidden = tf.layers.dense(inputs, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
    return {
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
        'belief': prior['belief'],
        'rnn_state': prior['rnn_state'],
    }
