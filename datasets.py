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

"""Wrapper for datasets."""

import functools
import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import datasets_utils
import glob
import cv2
import random
from absl import logging
import json

def resize_to_square(image, resolution=32, train=True):
  """Preprocess the image in a way that is OK for generative modeling."""

  # Crop a square-shaped image by shortening the longer side.
  image_shape = tf.shape(image)
  height, width, channels = image_shape[0], image_shape[1], image_shape[2]
  if height == width:
    side_size = resolution
  else:
    side_size = tf.minimum(height, width) 
  cropped_shape = tf.stack([side_size, side_size, channels])
  if train:
    image = tf.image.random_crop(image, cropped_shape)
  else:
    image = tf.image.resize_with_crop_or_pad(
        image, target_height=side_size, target_width=side_size)

  image = datasets_utils.change_resolution(image, res=resolution, method='area')
  return image

def ds_transform(example, con_factor=1.5, bright_factor=3):
  example['image'] = tf.cast(example['image'], tf.float32) / 255.0
  contrast_prob = tf.random.uniform([], 0, 1) < 0.5
  example['image'] = tf.cond(contrast_prob, lambda: tf.image.adjust_contrast(example['image'], con_factor), lambda: example['image'])
  example['image'] = tf.clip_by_value(example['image'], 0., 1.) * 255.
  example['image'] = tf.cast(example['image'], tf.uint16)
  brightness_prob = tf.random.uniform([], 0, 1) < 0.7
  example['image'] = tf.cond(brightness_prob, lambda: example['image'] * bright_factor, lambda: example['image'])
  example['image'] = tf.clip_by_value(example['image'], 0, 255)
  example['image'] = tf.cast(example['image'], tf.uint8)
  return example

def load_indices(json_filename):
  """
  input: json file that contains the appropriate index for every tile of the dataset
  returns a dictionary with the corresponding information
  """
  assert json_filename is not None
  with open(json_filename) as js:
    x = json.load(js)
  return x

def preprocess(example, train=True, resolution=256, n_channels=3):
  """Apply random crop (or) central crop to the image."""
  image = example

  is_label = False
  mask = None
  if isinstance(example, dict):
    image = example['image']
    is_label = 'label' in example.keys()
    mask = tf.cast(example['mask'], tf.uint8)

  if mask is not None:
    ind_mask = image.shape[-1]
    image = tf.concat((image, mask), axis=2)

  image = resize_to_square(image, train=train, resolution=resolution)

  # keepng 'file_name' key creates some undebuggable TPU Error.
  example_copy = dict()
  if mask is not None:
    image, mask = tf.split(image, [ind_mask, -1], axis=2)
    targets = image * tf.keras.backend.repeat_elements(mask, n_channels, axis=2)
    invmask = (1 - mask) * 40
    targets += tf.keras.backend.repeat_elements(invmask, n_channels, axis=2)
    # mask = tf.cast(mask, tf.bool)
    example_copy['mask'] = mask
  else:
    targets = image

  example_copy['image'] = image
  example_copy['targets'] = targets
  if is_label:
    example_copy['label'] = example['label']

  return example_copy


def get_gen_dataset(data_dir, batch_size, n_channels):
  """Converts a list of generated TFRecords into a TF Dataset."""

  def parse_example(example_proto, res=64):
    features = {'image': tf.io.FixedLenFeature([res*res*n_channels], tf.int64)}
    example = tf.io.parse_example(example_proto, features=features)
    image = tf.reshape(example['image'], (res, res, n_channels))
    return {'targets': image}

  # Provided generated dataset.
  def tf_record_name_to_num(x):
    x = x.split('.')[0]
    x = re.split(r'(\d+)', x)
    return int(x[1])

  assert data_dir is not None
  records = tf.io.gfile.listdir(data_dir)
  max_num = max(records, key=tf_record_name_to_num)
  max_num = tf_record_name_to_num(max_num)

  records = []
  for record in range(max_num + 1):
    path = os.path.join(data_dir, f'gen{record}.tfrecords')
    records.append(path)

  tf_dataset = tf.data.TFRecordDataset(records)
  tf_dataset = tf_dataset.map(parse_example, num_parallel_calls=100)
  tf_dataset = tf_dataset.batch(batch_size=batch_size)
  tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
  return tf_dataset

def create_gen_dataset_from_images(image_dir, mask_dir, config, train, subset):
  n_channels = config.get('n_channels', 3)
  flip_masks = config.get('flip_masks', False)
  # logging.info(f'Flipping masks')
  """Creates a dataset from the provided directory."""
  
  def categorize_mask(mask):
    mod_cover = config.get('max_coverage', 30)
    mask = cv2.imread(mask, 0)
    mask = 255 - mask if flip_masks else mask
    six = (mask.shape[0]-config.resolution[0])//2
    eix = (mask.shape[0]+config.resolution[0])//2
    siy = (mask.shape[1]-config.resolution[1])//2
    eiy = (mask.shape[1]+config.resolution[1])//2
    mask_val = mask[six:eix,siy:eiy]
    coverage = (np.count_nonzero(mask_val) / mask_val.size) * 100.0
    if coverage <= 5.0:
      category = 'clear'
    elif 5.0 < coverage <= mod_cover:
      category = 'moderate'
    elif coverage > mod_cover:
      category = 'severe'
    return category

  if config.get('jsonf', None) is not None:
    ref_dict = load_indices(config.get('jsonf', None))
  else:
    ref_dict = None

  mod_masks = []
  cube = []
  cloud_masks = []

  for mmask in sorted(glob.glob(mask_dir + "/*/**")):
    # create a list with moderately cloudy masks to use later
    category = categorize_mask(mmask)
    if category == 'moderate':
      mod_masks.append(mmask)
  # logging.info(f'Moderate masks found: {len(mod_masks)} ')

  for r, m in zip(sorted(glob.glob(image_dir + "/**")), sorted(glob.glob(mask_dir + "/**"))):
    for im, mask, ind in zip(sorted(glob.glob(r + "/**")), sorted(glob.glob(m + "/**")), enumerate(sorted(glob.glob(r + "/**")))):
      
      if ref_dict is not None:
        tilename = (os.path.basename(im).split("_"))[0]
        if subset == 'train':
          ref_index = int(ref_dict[tilename]['index'])
        elif subset == 'valid':
          ref_index = int(ref_dict[tilename]['val_index'])
        elif subset == 'test':
          ref_index = int(ref_dict[tilename]['test_index'])
        else:
          ref_index = None
      else:
        ref_index = config.ref_index
      assert ref_index is not None
      
      index = ind[0] # uncomment for both L2A datasets 
      # index = int(os.path.basename(im)[-6:-4]) # Uncomment for L1C
      
      # create the cube with the (T-4, ..., T-1) images masked with their own masks
      if ref_index-config.timeline+2 <= index < ref_index:
        im_mask = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        curr_mask = cv2.imread(mask, 0)
        curr_mask = 255 - curr_mask if flip_masks else curr_mask
        cloud_masks.append(curr_mask == 0)
        if n_channels == 3:
          im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2RGB)
        elif n_channels == 1:
          im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
        else:
          im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGRA2RGBA)
        cube.append(im_mask)  # encoder's input

      # add the last image to the cube list 2 times, with a random moderate mask + as is
      elif index == ref_index:
        im_mask = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        last_mask = mask
        curr_mask = cv2.imread(last_mask, 0)
        curr_mask = 255 - curr_mask if flip_masks else curr_mask
        cloud_masks.append(curr_mask == 0)

        # decoder's input
        last_clear = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        if n_channels == 3:
          last_clear = cv2.cvtColor(last_clear, cv2.COLOR_BGR2RGB)
        elif n_channels == 1:
          last_clear = cv2.cvtColor(last_clear, cv2.COLOR_BGR2GRAY)
        else:
          last_clear = cv2.cvtColor(last_clear, cv2.COLOR_BGRA2RGBA)
        cube.insert(0, last_clear)
        cloud_masks.insert(0, curr_mask == 0)

    # random masking on the last image
    if train:
      curr_mask = cv2.imread(random.choice(mod_masks), 0)
    elif subset=='valid':
      random.seed(int(os.path.basename(r)))
      curr_mask = cv2.imread(random.choice(mod_masks), 0)
    if subset !='test':
      curr_mask = 255 - curr_mask if flip_masks else curr_mask
      cloud_masks[-1] *= curr_mask == 0
    if n_channels == 3:
      im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2RGB)
    elif n_channels == 1:
      im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
    else:
      im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGRA2RGBA)
    cube.append(im_mask)  # encoder's input

    # apply random masking on every image during training
    if train:
      for i in range(len(cube) - 2):
        curr_mask = cv2.imread(random.choice(mod_masks), 0)
        curr_mask = 255 - curr_mask if flip_masks else curr_mask
        cloud_masks[i+1] *= curr_mask == 0

    sum_masks = np.sum(np.stack(cloud_masks, axis=2)[...,1:], axis=2)

    # if the last image is not clear skip the area
    last_category = categorize_mask(last_mask)
    if (subset != 'test' and last_category != 'clear' or np.any(sum_masks==0)) or \
              (subset == 'test' and last_category == 'clear' or np.any(sum_masks==0)):
      cube.clear()
      cloud_masks.clear()
      continue
    
    if n_channels == 1:
      files = tf.stack(cube, axis=-1)
    else: 
      files = tf.concat(cube, axis=2)  # creates tensors with size (256,256,T*3)
    cube_masks = tf.stack(cloud_masks, axis=2)

    yield {'image': files, 'mask': cube_masks}
    cube.clear()
    cloud_masks.clear()

def get_dataset(name,
                config,
                batch_size,
                subset,
                read_config=None,
                data_dir=None):
  """Wrapper around TF-Datasets.

  * Setting `config.random_channel to be True` adds
    ds['targets_slice'] - Channel picked at random. (of 3).
    ds['channel_index'] - Index of the randomly picked channel
  * Setting `config.downsample` to be True, adds:.
    ds['targets_64'] - Downsampled 64x64 input using tf.resize.
    ds['targets_64_up_back] - 'targets_64' upsampled using tf.resize

  Args:
    name: custom
    config: dict
    batch_size: batch size.
    subset: 'train', 'eval_train', 'valid' or 'test'.
    read_config: optional, tfds.ReadConfg instance. This is used for sharding
                 across multiple workers.
    data_dir: Data Directory, Used for Custom dataset.
  Returns:
   dataset: TF Dataset.
  """
  resolution = config.get('resolution', [256])
  n_channels = config.get('n_channels', 3)
  downsample = config.get('downsample', False)
  random_channel = config.get('random_channel', False)
  downsample_res = config.get('downsample_res', 64)
  downsample_method = config.get('downsample_method', 'area')
  num_epochs = config.get('num_epochs', -1)
  data_dir = config.get('data_dir') or data_dir
  mask_dir = config.get('mask_dir')
  target_dir = config.get('targets_dir', '')
  auto = tf.data.AUTOTUNE
  train = subset == 'train'
  augmentation = config.get('use_augmentation', False)

  #TODO: create different flags for each dataset version 
  if name == 'custom':
    assert data_dir is not None
    ds = tf.data.Dataset.from_generator(
            lambda: create_gen_dataset_from_images(data_dir, mask_dir, config, train=train, subset=subset),
            output_signature={'image': tf.TensorSpec(shape=(None,None,config.timeline*n_channels), dtype=tf.uint8),
                              'mask': tf.TensorSpec(shape=(None,None,config.timeline), dtype=tf.bool)}
    )
  else:
    raise ValueError(f'Expected dataset in [custom]. Got {name}')
  
  if train and augmentation:
      ds_transformed = ds.map(lambda x: ds_transform(x, 1.5, 3))
      ds = ds.concatenate(ds_transformed)

  ds = ds.map(lambda x: preprocess(x, train=train, resolution=resolution[0], n_channels=n_channels), num_parallel_calls=100)
  if train and random_channel:
    ds = ds.map(lambda x: datasets_utils.random_channel_slice(x, n_channels=n_channels))
  if downsample:
    downsample_part = functools.partial(
        datasets_utils.downsample_and_upsample,
        train=train,
        downsample_res=downsample_res,
        upsample_res=resolution[0],
        method=downsample_method)
    ds = ds.map(downsample_part, num_parallel_calls=100)
  if n_channels != 1:
    ds = ds.map(lambda x: datasets_utils.create_grayscale_cubes(x, down_res=downsample_res, n_channels=n_channels, downsample=downsample)) #TODO: check if this is needed

  if target_dir:
    # it = iter(ds)
    # cnt = 0
    # for data in it:
    #   cnt += 1
    # logging.info(f'Total number: {cnt}')
    datasets_utils.save_dataset(ds, target_dir, subset, downsample_res, n_channels)
  
  if train:
    ds = ds.shuffle(buffer_size=128, reshuffle_each_iteration=True)
    ds = ds.repeat(num_epochs)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(auto)
  logging.info(ds)
  return ds
