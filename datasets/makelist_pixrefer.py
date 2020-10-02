import numpy as np
import os
from optparse import OptionParser
import json
import logging
import sys

sys.path.append(os.getcwd())
from config.configure import YParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def write_dataset(params):
  train_dataset_path = params.train_dataset_path
  eval_dataset_path = params.eval_dataset_path
  root_path = params.root_path
  train_by_eval = params.train_by_eval

  sample_index = 0

  with open(train_dataset_path, "w") as train_file:
    with open(eval_dataset_path, "w") as eval_file:
      for root, subdirs, files in os.walk(root_path):
        if not subdirs:
            logger.info('Processing {}'.format(root))
            count = 0
            for file in files:
              if (file.endswith('.jpg')):
                count += 1

            sample_index += 1
            if (sample_index % (train_by_eval + 1) == 0):
              eval_file.write("{}|{}\n".format(root, count))
            else:
              train_file.write("{}|{}\n".format(root, count))


if (__name__ == '__main__'):
  cmd_parser = OptionParser(usage="usage: %prog [options] --config_path <>")
  cmd_parser.add_option('--config_path', type="string", dest="config_path",
                        help='the config json file')

  opts, argv = cmd_parser.parse_args()

  if (not opts.config_path is None):
    config_path = opts.config_path

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists')
      exit(0)

    params = YParams(config_path, 'default')
    write_dataset(params)
  else:
    print('Please check your parameters.')
