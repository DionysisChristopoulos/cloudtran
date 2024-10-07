from absl import app
from absl import flags
from absl import logging

import os, glob
import random
import shutil

flags.DEFINE_string('sample_dir', './Datasets/Final_Checkpoints/sup_512_conv_sen12ms_L2Amerged_50cov_noaug/samples_sup_test_on_L1C_parallel', 'Directory where samples are saved.')
flags.DEFINE_string('output_dir', './Datasets/selected_samples_journal/inference_L1CBYL2A_sup_parallel', 'Directory where the selected samples will be copied.')
flags.DEFINE_float('pick_percentage', 0.2, 'Percentage of the total samples to be selected')
FLAGS = flags.FLAGS

def random_index_gen(sample_dir: str, pick_percentage: float = 0.2):
    total_samples = (len(os.listdir(sample_dir)) - 1) // 5
    indices = list(range(total_samples))
    random.seed(23)
    random_indices = random.sample(indices, int(pick_percentage*total_samples))
    return random_indices

def main(_):
    sample_dir = FLAGS.sample_dir
    pick_perc = FLAGS.pick_percentage
    
    logging.info(f'Randomly picking samples from {sample_dir}.')
    idxs = random_index_gen(sample_dir=sample_dir, pick_percentage=pick_perc)
    logging.info(f'Successfully picked {len(idxs)} random samples.')

    out_dir = FLAGS.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for idx in idxs:
        images = glob.glob(os.path.join(sample_dir, f'{str(idx).zfill(5)}*.png'))
        assert len(images) == 5
        for img in images:
            shutil.copy(img, out_dir)
    logging.info(f'Successfully copied selected samples!')

if __name__ == '__main__':
  app.run(main)