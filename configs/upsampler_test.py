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

"""Test configurations for color upsampler."""
from ml_collections import ConfigDict
from os import getenv

resolution = int(getenv("DOWNSAMPLE_SIZE", 64))
model_size = int(getenv("MODEL_SIZE", 512))
num_channels = int(getenv("NUM_CHANNELS", 3))

def get_config():
  """Experiment configuration."""
  config = ConfigDict()

  # Data.
  config.dataset = 'custom'
  config.downsample = True
  config.random_channel = False
  config.downsample_res = resolution
  config.resolution = [224, 224]
  config.n_channels = num_channels
  config.timeline = 6
  config.ref_index = 55
  config.jsonf = None
  config.max_coverage = 30
  config.mask_dir = './Datasets/Timeseries_cropped_512/masks_final_testset'
  config.data_dir = './Datasets/Timeseries_cropped_512/videos_final_testset'
  # config.targets_dir = './Datasets/cloudtran_sup_out'
  config.flip_masks = False
  config.use_discriminator = False # always False during evaluation/validation/inference

  # Training.
  config.batch_size = 1
  config.max_train_steps = 50000
  config.save_checkpoint_secs = 900
  config.num_epochs = -1
  config.polyak_decay = 0.999
  config.eval_num_examples = 100
  config.eval_batch_size = 1
  config.eval_checkpoint_wait_secs = -1

  config.optimizer = ConfigDict()
  config.optimizer.type = 'rmsprop'
  config.optimizer.learning_rate = 3e-4

  print("Model size: {}".format(model_size))
  # Model.
  config.model = ConfigDict()
  config.model.n_channels = num_channels
  config.model.hidden_size = model_size
  config.model.ff_size = model_size
  config.model.num_heads = 4
  config.model.num_encoder_layers = 3
  config.model.resolution = [resolution, resolution]
  config.model.name = 'upsampler'

  config.sample = ConfigDict()
  config.sample.n_channels = num_channels
  config.sample.gen_data_dir = './path_to_cloudtran_core_logdir/samples_core_valid'
  config.sample.log_dir = 'samples_sup_valid'
  config.sample.batch_size = 5
  config.sample.mode = 'argmax'
  config.sample.num_samples = 1
  config.sample.num_outputs = 100
  config.sample.skip_batches = 0
  config.sample.gen_file = 'gen0_sup_valid'
  config.sample.im_outputs = True
  config.sample.upsample_factor = 1

  return config
