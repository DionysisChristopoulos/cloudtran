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

"""Train configurations for CloudTran Core."""
from ml_collections import ConfigDict
from os import getenv

resolution = int(getenv("DOWNSAMPLE_SIZE", 64))
model_size = int(getenv("MODEL_SIZE", 128))
num_channels = int(getenv("NUM_CHANNELS", 3))

def get_config():
  """Experiment configuration."""
  config = ConfigDict()

  # Data.
  config.dataset = 'custom'
  config.downsample = True
  config.random_channel = True
  config.downsample_res = resolution
  config.resolution = [224, 224]
  config.n_channels = num_channels
  config.timeline = 6 # number of dates to include in training + 1 for the GT target image
  config.ref_index = 55 # index to specify the clean target image.
  config.jsonf = None # Path to specify json file for adaptive indexing. If None ref_index is used.
  config.max_coverage = 30
  config.mask_dir = './Datasets/Timeseries_cropped_512/masks_final_trainset'
  config.data_dir = './Datasets/Timeseries_cropped_512/videos_final_trainset'
  # config.targets_dir = './Datasets/cloudtran_core_out' # path to save the training input time-series
  config.flip_masks = False
  config.use_discriminator = True
  config.use_augmentation = False

  # Training.
  config.batch_size = 1
  config.max_train_steps = 50000
  config.save_checkpoint_secs = 900
  config.num_epochs = -1
  config.polyak_decay = 0.999
  config.eval_num_examples = 20000
  config.eval_batch_size = 16
  config.eval_checkpoint_wait_secs = -1

  # loss hparams.
  config.loss_factor = 0.99
  config.encoder_loss_factor = 0.01
  config.generator_loss_factor = 0.01

  config.optimizer = ConfigDict()
  config.optimizer.type = 'rmsprop'
  config.optimizer.learning_rate = 3e-4

  print("Model size: {}".format(model_size))
  # Model.
  config.model = ConfigDict()
  config.model.n_channels = num_channels
  config.model.hidden_size = model_size
  config.model.stage = 'encoder_decoder'
  config.model.resolution = [resolution, resolution]
  config.model.name = 'cloudtran_core'

  # encoder
  config.model.encoder = ConfigDict()
  config.model.encoder.n_channels = num_channels
  config.model.encoder.ff_size = model_size
  config.model.encoder.hidden_size = model_size
  config.model.encoder.num_heads = 4
  config.model.encoder.num_encoder_layers = 4
  config.model.encoder.num_temp_layers = 2
  config.model.encoder.dropout = 0.0
  config.model.encoder.posemb = 'learnable'
  config.model.encoder.aggregation = 'conv'

  # decoder
  config.model.decoder = ConfigDict()
  config.model.decoder.n_channels = num_channels
  config.model.decoder.ff_size = model_size
  config.model.decoder.hidden_size = model_size
  config.model.decoder.resolution = [resolution, resolution]
  config.model.decoder.num_heads = 4
  config.model.decoder.num_inner_layers = 2
  config.model.decoder.num_outer_layers = 2
  config.model.decoder.dropout = 0.0
  config.model.decoder.skip = True

  config.model.decoder.cond_mlp = 'affine'
  config.model.decoder.cond_mlp_act = 'identity'

  config.model.decoder.cond_ln_act = 'identity'
  config.model.decoder.cond_ln = True
  config.model.decoder.cond_ln_seq = 'sc'
  config.model.decoder.cond_ln_sp_ave = 'learnable'
  config.model.decoder.cond_ln_init = 'glorot_uniform'

  config.model.decoder.cond_att_init = 'glorot_uniform'
  config.model.decoder.cond_att_v = True
  config.model.decoder.cond_att_q = True
  config.model.decoder.cond_att_k = True
  config.model.decoder.cond_att_scale = True
  config.model.decoder.cond_att_act = 'identity'
  return config
