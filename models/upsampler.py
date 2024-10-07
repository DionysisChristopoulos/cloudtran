# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

"""Upsampler model of CloudTran.

Upsamples a 256x256 blurry low resolution image into the final 256x256 high resolution output.

"""
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers, Sequential
from models import layers as tran_layers
from utils import base_utils


class Upsampler(tf.keras.Model):
  """Spatial Upsampler."""

  def __init__(self, config, **kwargs):
    super(Upsampler, self).__init__(**kwargs)
    self.config = config
    self.num_symbols = 256
    self.hidden_size = self.config.get('hidden_size', 512)
    self.down_res = self.config.get('down_res', 32)
    self.down_method = self.config.get('down_method', 'area')

  def build(self, input_shape):
    self.channel_embedding = layers.Dense(
        units=self.hidden_size, use_bias=False)
    self.timecube_embedding = layers.Dense(
        units=self.hidden_size, use_bias=False)
    self.input_dense = layers.Dense(units=self.hidden_size)
    self.encoder = tran_layers.FactorizedAttention(self.config)
    self.final_dense = layers.Dense(units=self.num_symbols)

  def call(self, inputs, inputs_slice, channel_index=None, training=True):
    """Super resolves blurry high resolution inputs into per-pixel logits.

    Args:
      inputs: size (B, H, W, num_channels*timeline).
      inputs_slice: batch of randomly sliced channels, i.e (B, H, W, 1)
                    each element of the batch is either a R, G or B channel.
      channel_index: size (B,) Each element is (0, 1, or 2) denoting a
                     R, G or B channel.
      training: used only for dropout.
    Returns:
      logits: size (B, H, W, num_channels, num_symbols) during training or
              size (B, H, W, 1, num_symbols) during evaluation or sampling.
    """
    if channel_index is not None:
      up_input = inputs_slice[..., :1]
    else:
      up_input = inputs_slice[..., :3]
    
    # FIXME: hard-coded indexing
    timecube = inputs[:, :, :, 3:] # [B, 256, 256, channels*(timeline-1)]

    logits = self.upsampler(up_input, timecube, training=training, 
                            channel_index=channel_index)
    return logits, {}

  def upsampler(self, inputs, timecube, channel_index=None, training=True):
    batch, height, width, num_channels = inputs.shape
    logits = []

    timecube = tf.one_hot(timecube, depth=self.num_symbols)  # (B, 256, 256, 15, 256)
    timecube = tf.reshape(timecube, [batch, height, width, 1, -1])  # (B, 256, 256, 1, 15*256)
    time_embed = self.timecube_embedding(timecube)  # (B, 256, 256, 1, hs)
    time_embed = tf.squeeze(time_embed, axis=-2)  # (B, 256, 256, hs)

    if channel_index is not None:
      channel_index = tf.reshape(channel_index, (-1, 1, 1))

    for channel_ind in range(num_channels):
      channel = inputs[Ellipsis, channel_ind]

      if channel_index is not None:
        # single random channel slice during training.
        # channel_index is the index of the random channel.
        channel += self.num_symbols * channel_index
      else:
        channel += self.num_symbols * channel_ind

      channel = tf.expand_dims(channel, axis=-1)
      channel = tf.one_hot(channel, depth=self.num_symbols*3)

      channel = self.channel_embedding(channel)
      channel = tf.squeeze(channel, axis=-2)

      channel = tf.concat((channel, time_embed), axis=-1)
      channel = self.input_dense(channel)

      context = self.encoder(channel, training=training)
      channel_logits = self.final_dense(context)
      logits.append(channel_logits)
    logits = tf.stack(logits, axis=-2)
    return logits

  def sample(self, time_cond, inputs, mode='argmax'):
    output = dict()
    output['low_res_cond'] = tf.cast(inputs, dtype=tf.uint8)
    time_cond = time_cond[:, :, :, 3:]  # FIXME: hard-coded indexing

    logits = self.upsampler(inputs, time_cond, training=False)

    if mode == 'argmax':
      samples = tf.argmax(logits, axis=-1)
    elif mode == 'sample':
      batch_size, height, width, channels = logits.shape[:-1]
      logits = tf.reshape(logits, (batch_size*height*width*channels, -1))
      samples = tf.random.categorical(logits, num_samples=1,
                                      dtype=tf.int32)[:, 0]
      samples = tf.reshape(samples, (batch_size, height, width, channels))

    samples = tf.cast(samples, dtype=tf.uint8)
    output[f'high_res_{mode}'] = samples
    return output

  @property
  def metric_keys(self):
    return []

  def loss(self, targets, logits, train_config, training, aux_output=None):
    if training:
      labels = targets['targets_slice']
    else:
      labels = targets['targets']

    c = 1 if training else 3 #TODO: Get 3 through num_channels from train_config
    labels = labels[:, :, :, :c]

    height, width, num_channels = labels.shape[1:]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    loss = loss / (height * width * num_channels)
    return loss, {}

  def get_logits(self, inputs_dict, train_config, training):
    downsample_res = train_config.get('downsample_res', 64)

    # Random channel slice during training denoted by suffix 'slice'.
    # up_back suffix denotes blurry high resolution input.
    inputs = inputs_dict['targets']
    if training:
      inputs_slice = inputs_dict['targets_slice_%d_up_back' % downsample_res]
      channel_index = inputs_dict['channel_index']
    else:
      inputs_slice = inputs_dict['targets_%d_up_back' % downsample_res]
      channel_index = None
    return self.call(
        inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)
