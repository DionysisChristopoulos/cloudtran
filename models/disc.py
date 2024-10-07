import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime, time
from keras.constraints import Constraint
from keras import backend as K

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('chkpt_dir', "./Datasets/Discriminator/training_checkpoints_exp_2", 'Where checkpoints will be saved.')
flags.DEFINE_string('logging_dir', "./Datasets/Discriminator/logs/", 'Where logs will be saved.')
flags.DEFINE_string('gendir', "./Datasets/disc_inputs/generated_valid_noisy", 'Directory with generated images.')
flags.DEFINE_string('unmdir', "./Datasets/disc_inputs/unmasked_valid_noisy", 'Directory with unmasked images.')
flags.DEFINE_string('mdir', "./Datasets/disc_inputs/masked_valid_noisy", 'Directory with masked images.')
flags.DEFINE_integer('res', 64, 'Image resolution')
FLAGS = flags.FLAGS

def get_disc_dataset(gen_dir, unm_dir, m_dir):
    gen_ds = tf.keras.utils.image_dataset_from_directory(gen_dir, labels=None, color_mode='rgb', batch_size=1, image_size=(FLAGS.res, FLAGS.res), shuffle=False)
    unm_ds = tf.keras.utils.image_dataset_from_directory(unm_dir, labels=None, color_mode='rgb', batch_size=1, image_size=(FLAGS.res, FLAGS.res), shuffle=False)
    m_ds = tf.keras.utils.image_dataset_from_directory(m_dir, labels=None, color_mode='rgb', batch_size=1, image_size=(FLAGS.res, FLAGS.res), shuffle=False)

    disc_ds = tf.data.Dataset.zip((gen_ds, unm_ds, m_ds))
    disc_ds.shuffle(225, reshuffle_each_iteration=True)
    disc_ds.batch(16)
    return disc_ds

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

def downsample(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    const = ClipConstraint(1)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False, kernel_constraint=const))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    const = ClipConstraint(1)
    
    masked = tf.keras.layers.Input(shape=[FLAGS.res, FLAGS.res, 1], name='masked_image')
    tar = tf.keras.layers.Input(shape=[FLAGS.res, FLAGS.res, 1], name='target_image')
    ch_ind = tf.keras.layers.Input(shape=[1], name='channel')
    ch_ind_reshaped = tf.reshape(ch_ind, [-1, 1, 1, 1])

    cast_layer = tf.keras.layers.Lambda(lambda x: K.tf.cast(x, 'float32'))
    div_layer = tf.keras.layers.Lambda(lambda inputs: K.tf.math.divide(cast_layer(inputs[0]), cast_layer(inputs[1])))
    sin_layer = tf.keras.layers.Lambda(lambda x: K.tf.math.sin(x))
    cos_layer = tf.keras.layers.Lambda(lambda x: K.tf.math.cos(x))
    add_layer = tf.keras.layers.Lambda(lambda inputs: inputs[0] + inputs[1])
    sub_layer = tf.keras.layers.Lambda(lambda inputs: inputs[0] - inputs[1])

    mask_emb = add_layer([masked, 256*ch_ind_reshaped])
    tar_emb = add_layer([tar, 256*ch_ind_reshaped])

    x = tf.keras.layers.concatenate([mask_emb, tar_emb])  # (batch_size, 64, 64, channels*2)

    down1 = downsample(64, 4, strides=1, apply_batchnorm=False)(x)  # (batch_size, 64, 64, 64)
    down2 = downsample(128, 4, strides=1)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4, strides=2)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False, kernel_constraint=const)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, kernel_constraint=const)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[masked, tar, ch_ind], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss, real_loss, generated_loss

@tf.function
def train_step(generated, unmasked, masked, random_channel, step, discriminator, optimizer, summary_writer):
    
    with tf.GradientTape() as disc_tape:
        random_channel_ind = tf.expand_dims(random_channel, 0)
        disc_real_output = discriminator([masked, unmasked, random_channel_ind], training=True)
        disc_generated_output = discriminator([masked, generated, random_channel_ind], training=True)

        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('disc_loss', disc_loss, step=step//100)

def fit(train_ds, steps):
    
    discriminator = Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    checkpoint_dir = FLAGS.chkpt_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)

    log_dir=FLAGS.logging_dir
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    start = time.time()

    for step, element in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()
            print(f"Step: {step//1000}k")

        random_channel = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        train_step(element[0][Ellipsis, random_channel], element[1][Ellipsis, random_channel], element[2][Ellipsis, random_channel], random_channel, step, discriminator, discriminator_optimizer, summary_writer)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 20000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

def main(_):
    train_dataset = get_disc_dataset(FLAGS.gendir, FLAGS.unmdir, FLAGS.mdir)
    fit(train_dataset, steps=100000)

if __name__ == '__main__':
    app.run(main)