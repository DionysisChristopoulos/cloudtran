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

"""Utils for datasets loading."""

import os
import tensorflow as tf

from PIL import Image
import cv2
import matplotlib.pyplot as plt

def plot_cube(cube, n_ch):
    n_im = cube.shape[-1]//n_ch
    for cnt, i in enumerate(range(0,n_ch*n_im,n_ch)):
        plt.subplot(1,n_im+1,cnt+1)
        plt.imshow(cube[...,i:i+n_ch])
    plt.show()

def change_resolution(image, res, method='area'):
  image = tf.image.resize(image, method=method, antialias=True,
                          size=(res, res))
  image = tf.cast(tf.round(image), dtype=tf.int32)
  return image


def downsample_and_upsample(x, train, downsample_res, upsample_res, method):
  """Downsample and upsample."""
  keys = ['targets']
  if 'mask' in x.keys():
    keys += ['mask']
    keys += ['image']
  if train and 'targets_slice' in x.keys():
    keys += ['targets_slice']

  for key in keys:
    inputs = x[key]
    # Conditional low resolution input.
    x_down = change_resolution(inputs, res=downsample_res, method=method)
    x['%s_%d' % (key, downsample_res)] = x_down

    # We upsample here instead of in the model code because some upsampling
    # methods are not TPU friendly.
    x_up = change_resolution(x_down, res=upsample_res, method=method)
    x['%s_%d_up_back' % (key, downsample_res)] = x_up
  return x

def gray_conversion(tensor):
  array = tensor.numpy()
  array = cv2.cvtColor(array.astype('uint8'), cv2.COLOR_BGR2GRAY)
  array = array.astype('int32')
  return array

def create_grayscale_cubes(x, down_res, n_channels=3, downsample=True):
  keys = ['targets', 'image']
  if downsample:
    keys += [f'targets_{down_res}', f'targets_{down_res}_up_back', f'image_{down_res}', f'image_{down_res}_up_back']
  for key in keys:
    res = x[key].shape[1]
    ch = x[key].shape[2]
    gray_list = []
    for index in range(ch//n_channels):
      gray_key = x[key][Ellipsis, index*n_channels:index*n_channels+n_channels]
      gray_key = tf.py_function(gray_conversion, [gray_key], tf.int32)
      gray_list.append(gray_key)
    gray_cube = tf.stack(gray_list, axis=-1)
    gray_cube = tf.convert_to_tensor(gray_cube)
    gray_cube.set_shape([res, res, ch//n_channels])
    x[f'{key}_grayscale'] = gray_cube
  return x

def random_channel_slice(x, n_channels=3):
  random_channel = tf.random.uniform(
      shape=[], minval=0, maxval=n_channels, dtype=tf.int32)
  targets = x['targets']
  res = targets.shape[1]
  ch = targets.shape[2]
  image_slice = targets[Ellipsis, random_channel::n_channels]
  image_slice.set_shape([res, res, ch//n_channels])
  x['targets_slice'] = image_slice
  x['channel_index'] = random_channel
  return x

def save_dataset(ds, target_path, subset, downsample_res=64, n_channels=3):
  
  # save each clear and random cloudy image in target path for evaluation
  # target_path = config.get('targets_dir')
  target_path = target_path+f"_{downsample_res}"+f"_{n_channels}bands"
  if not os.path.exists(target_path):
    os.makedirs(target_path)
  
  save_dir = os.path.join(target_path, subset)
  image_dir = os.path.join(save_dir, 'image')
  mask_dir = os.path.join(save_dir, 'mask')
  if not os.path.exists(save_dir):
      os.makedirs(image_dir)
      os.makedirs(mask_dir)
  for cnt, element in enumerate(ds.as_numpy_iterator()):
    n_im = element['image'].shape[-1]//n_channels
    eldirim = os.path.join(image_dir, f'{cnt:05d}')
    eldirmask = os.path.join(mask_dir, f'{cnt:05d}')
    if not os.path.exists(eldirim):
      os.makedirs(eldirim)
      os.makedirs(eldirmask, exist_ok=True)
    
    for i in range(1, n_im):
      image = element[f'image_{downsample_res}'][:, :, i*n_channels:(i+1)*n_channels].astype('uint8')
      mask = element[f'mask_{downsample_res}'][:, :, i].astype('uint8')*255

      if n_channels==3:
        final_im = Image.fromarray(image, mode='RGB')
      elif n_channels==1:
        image = image.squeeze(axis=-1)
        final_im = Image.fromarray(image, mode='L')
      else:
        final_im = Image.fromarray(image, mode='RGBA')
      final_mask = Image.fromarray(mask, mode='L')
      final_im.save(os.path.join(eldirim, f'{i:03d}.png'))
      final_mask.save(os.path.join(eldirmask, f'{i:03d}.png'))

