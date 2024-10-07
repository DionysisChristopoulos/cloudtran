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
"""CloudTran core.

Core autoregressive component of CloudTran based on
the AxialTransformer with conditional self-attention layers.

"""
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from models import core
from models import layers as tran_layers
from utils import base_utils

class CloudTranCore(tf.keras.Model):
  """Core Transformer."""

  def __init__(self, config, **kwargs):
    super(CloudTranCore, self).__init__(**kwargs)
    self.config = config

    # 3 bits per channel, 8 colors per channel, a total of 512 colors.
    self.num_symbols = 256
    self.num_symbols_flat = 256 * config.get('n_channels', 3)

    self.enc_cfg = config.encoder
    self.dec_cfg = config.decoder
    self.hidden_size = self.config.get('hidden_size',
                                       self.dec_cfg.hidden_size)

    # stage can be 'encoder_decoder' or 'decoder'
    # 1. decoder -> loss only due to autoregressive model.
    # 2. encoder_decoder -> loss due to both the autoregressive and parallel
    # model.
    # encoder_only and all
    self.stage = config.get('stage', 'decoder')
    self.is_parallel_loss = 'encoder' in self.stage
    stages = ['decoder', 'encoder_decoder']
    if self.stage not in stages:
      raise ValueError('Expected stage to be in %s, got %s' %
                       (str(stages), self.stage))

  @property
  def metric_keys(self):
    if self.stage == 'encoder_decoder':
      return ['encoder']
    return []

  def build(self, input_shape):
    # encoder graph
    self.encoder = core.TimeseriesEncoder(self.enc_cfg)
    if self.is_parallel_loss:
      self.parallel_dense = layers.Dense(
          units=self.num_symbols, name='parallel_logits', use_bias=False)

    # decoder graph: outer decoder -> inner decoder -> logits.
    self.pixel_embed_layer = layers.Dense(units=self.hidden_size, use_bias=False)
    self.outer_decoder = core.OuterDecoder(self.dec_cfg)
    self.inner_decoder = core.InnerDecoder(self.dec_cfg)
    self.final_dense = layers.Dense(units=self.num_symbols, name='auto_logits')
    self.final_norm = layers.LayerNormalization()

  def call(self, inputs, inputs_slice, channel_index=None, training=True):
    # encodes timeseries cube (H, W, C*T) into activations of shape (H, W, model_dim).
    if channel_index is not None:
      dec_inputs, enc_inputs = tf.split(inputs_slice, [1,-1], axis=3)
    else:
      dec_inputs, enc_inputs = tf.split(inputs_slice, [self.config.get('n_channels', 3),-1], axis=3)
    z = self.encoder(enc_inputs, channel_index=channel_index, training=training)

    if self.is_parallel_loss:
      enc_logits = self.parallel_dense(z)
      # enc_logits = tf.expand_dims(enc_logits, axis=-2)
      # (1, 64, 64, 1, model_dim)

    dec_logits = self.decoder(dec_inputs, z, channel_index=channel_index, training=training)
    if self.is_parallel_loss:
      return dec_logits, {'encoder_logits': enc_logits}
    return dec_logits, {}

  def decoder(self, inputs, z, channel_index=None, training=True):
    """Decodes timeseries representation and masked context into logits."""
    num_channels = inputs.shape[-1]
    logits = []
    
    if channel_index is not None:
        channel_index = tf.reshape(channel_index, (-1, 1, 1))

    for channel_ind in range(num_channels):
      channel = inputs[Ellipsis, channel_ind]

      if channel_index is not None:
          # single random channel slice during training.
          # channel_index is the index of the random channel.
          # each channel has 8 possible symbols.
          channel += 256 * channel_index
      else:
          channel += 256 * channel_ind

      channel = tf.expand_dims(channel, axis=-1)
      channel = tf.one_hot(channel, depth=self.num_symbols_flat)

      channel = self.pixel_embed_layer(channel)
      h_dec = tf.squeeze(channel, axis=-2)

      h_upper = self.outer_decoder((h_dec, z[...,channel_ind,:]), training=training)
      h_inner = self.inner_decoder((h_dec, h_upper, z[...,channel_ind,:]), training=training)

      activations = self.final_norm(h_inner)
      logits.append(self.final_dense(activations))
    logits = tf.stack(logits, axis=-2)
    return logits

  def image_loss(self, logits, labels):
    """Cross-entropy between the logits and labels."""
    height, width, num_channel = labels.shape[1:4]
    # logits = tf.squeeze(logits, axis=-2)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    return loss / (height * width * num_channel)

  def loss(self, targets, logits, train_config, training, aux_output=None):
    """Converts targets to coarse colors and computes log-likelihood."""
    is_downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    n_channels = train_config.get('n_channels', 3)
    if is_downsample and training:
      labels = targets['targets_slice_%d' % downsample_res]
    elif is_downsample:
      labels = targets['targets_%d' % downsample_res]
    elif training:
      labels = targets['targets_slice']
    else:
      labels = targets['targets']

    if aux_output is None:
      aux_output = {}

    c = 1 if training else n_channels
    labels = labels[:, :, :, :c]

    loss = self.image_loss(logits, labels)
    enc_logits = aux_output.get('encoder_logits')
    if enc_logits is None:
      return loss, {}

    enc_loss = self.image_loss(enc_logits, labels)
    return loss, {'encoder': enc_loss}

  def get_logits(self, inputs_dict, train_config, training):
    is_downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    channel_index = inputs_dict['channel_index'] if training else None
    inputs_key = 'targets_%d' % downsample_res if is_downsample else 'targets'
    inputs = inputs_dict[inputs_key]
    if is_downsample and training:
      inputs_slice = inputs_dict['targets_slice_%d' % downsample_res]
    elif is_downsample:
      inputs_slice = inputs_dict['targets_%d' % downsample_res]
    elif training:
      inputs_slice = inputs_dict['targets_slice']
    else:
      inputs_slice = inputs_dict['targets']

    return self(inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)
  
  def sample(self, time_cond, mode='argmax', only_parallel=False):
    output = {}
    n_channels = self.config.get('n_channels', 3)

    z_timecube = self.encoder(time_cond[:, :, :, n_channels:], training=False)
    if self.is_parallel_loss:
      z_logits = self.parallel_dense(z_timecube)
      parallel_image = tf.argmax(z_logits, axis=-1, output_type=tf.int32)
      # parallel_image = self.post_process_image(parallel_image)

      output['parallel'] = parallel_image
      if only_parallel:
        return output
    
    ch_context = z_timecube
    images = []
    probas = []
    for ch in range(n_channels):
      image, proba = self.autoregressive_sample(z_timecube=ch_context[...,ch,:], channel=ch, mode=mode)
      # ch_context += self.encoder(image, channel_index=ch, training=False)
      images.append(image)
      probas.append(proba)
    output['auto_%s' % mode] = tf.stack(images,axis=-1)
    output['proba'] = tf.stack(probas,axis=-1)
    return output

  def autoregressive_sample(self, z_timecube, channel, mode='sample'):
    """Generates pixel-by-pixel.

    1. The encoder is run once per-channel.
    2. The outer decoder is run once per-row.
    3. the inner decoder is run once per-pixel.

    The context from the encoder and outer decoder conditions the
    inner decoder. The inner decoder then generates a row, one pixel at a time.

    After generating all pixels in a row, the outer decoder is run to recompute
    context. This condtions the inner decoder, which then generates the next
    row, pixel-by-pixel.

    Args:
      z_timecube: timeseries cube context, comes from encoder.
      mode: sample or argmax.

    Returns:
      image: coarse image of shape (B, H, W)
      image_proba: probalities, shape (B, H, W, 512)
    """
    num_filters = self.config.hidden_size
    batch_size, height, width = z_timecube.shape[:3]

    # channel_cache[i, j] stores the pixel embedding for row i and col j.
    canvas_shape = (batch_size, height, width, num_filters)
    channel_cache = tran_layers.Cache(canvas_shape=(height, width))
    init_channel = tf.zeros(shape=canvas_shape) 
    init_ind = tf.stack([0, 0])
    channel_cache(inputs=(init_channel, init_ind))

    # upper_context[row_ind] stores context from all previously generated rows.
    upper_context = tf.zeros(shape=canvas_shape)

    # row_cache[0, j] stores the pixel embedding for the column j of the row
    # under generation. After every row is generated, this is rewritten.
    row_cache = tran_layers.Cache(canvas_shape=(1, width))
    init_row = tf.zeros(shape=(batch_size, 1, width, num_filters))
    row_cache(inputs=(init_row, init_ind))

    pixel_samples, pixel_probas = [], []

    for row in range(height):
      row_cond_channel = tf.expand_dims(z_timecube[:, row], axis=1)
      row_cond_upper = tf.expand_dims(upper_context[:, row], axis=1)
      row_cache.reset()

      gen_row, proba_row = [], []
      for col in range(width):

        inner_input = (row_cache.cache, row_cond_upper, row_cond_channel)
        # computes output activations at col.
        activations = self.inner_decoder(inner_input, row_ind=row,
                                          training=False)

        pixel_sample, pixel_embed, pixel_proba = self.act_logit_sample_embed(
            activations, col, channel, mode=mode)
        proba_row.append(pixel_proba)
        gen_row.append(pixel_sample)

        # row_cache[:, col] = pixel_embed
        row_cache(inputs=(pixel_embed, tf.stack([0, col])))

        # channel_cache[row, col] = pixel_embed
        channel_cache(inputs=(pixel_embed, tf.stack([row, col])))

      gen_row = tf.stack(gen_row, axis=-1)
      pixel_samples.append(gen_row)
      pixel_probas.append(tf.stack(proba_row, axis=1))

      # after a row is generated, recomputes the context for the next row.
      # upper_context[row] = self_attention(channel_cache[:row_index])
      upper_context = self.outer_decoder(
          inputs=(channel_cache.cache, z_timecube), training=False)

    image = tf.stack(pixel_samples, axis=1)
    # image = self.post_process_image(image)

    image_proba = tf.stack(pixel_probas, axis=1)
    return image, image_proba

  def act_logit_sample_embed(self, activations, col_ind, channel, mode='sample'):
    """Converts activations[col_ind] to the output pixel.

    Activation -> Logit -> Sample -> Embedding.

    Args:
      activations: 5-D Tensor, shape=(batch_size, 1, width, hidden_size)
      col_ind: integer.
      mode: 'sample' or 'argmax'
    Returns:
      pixel_sample: 1-D Tensor, shape=(batch_size, 1, 1)
      pixel_embed: 4-D Tensor, shape=(batch_size, 1, 1, hidden_size)
      pixel_proba: 4-D Tensor, shape=(batch_size, 1, 512)
    """
    batch_size = activations.shape[0]
    pixel_activation = tf.expand_dims(activations[:, :, col_ind], axis=-2)
    pixel_logits = self.final_dense(self.final_norm(pixel_activation))
    pixel_logits = tf.squeeze(pixel_logits, axis=[1, 2])
    pixel_proba = tf.nn.softmax(pixel_logits, axis=-1)

    if mode == 'sample':
      pixel_sample = tf.random.categorical(
          pixel_logits, num_samples=1, dtype=tf.int32)
      pixel_sample = tf.squeeze(pixel_sample, axis=-1)
    elif mode == 'argmax':
      pixel_sample = tf.argmax(pixel_logits, axis=-1, output_type=tf.int32)

    pixel_sample_expand = tf.reshape(pixel_sample+256*channel, [batch_size, 1, 1])
    pixel_one_hot = tf.one_hot(pixel_sample_expand, depth=self.num_symbols_flat)
    pixel_embed = self.pixel_embed_layer(pixel_one_hot)
    return pixel_sample, pixel_embed, pixel_proba

  def post_process_image(self, image):
    """Post process image of size (H, W, 512) to a coarse RGB image."""
    image = base_utils.bins_to_labels(
        image, num_symbols_per_channel=self.num_symbols_per_channel)
    image = base_utils.convert_bits(image, n_bits_in=3, n_bits_out=8)
    image = tf.cast(image, dtype=tf.uint8)
    return image
