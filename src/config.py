#coding:utf-8
author__='zhiyu yin'

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

"""cfg变量中保存的是默认config
ymal文件保存变动性config
"""
__C = edict()
cfg = __C  # 引用传递

__C.NUM_CLASSES = 30
__C.GPUS = '1'  # 末尾无逗号
__C.SUMMARY_DIR = '/data/yinzhiyu/results/Graph-Memory-Networks/log'

#------关于输入的配置-------#
__C.INPUT = edict()
__C.INPUT.QUERY_SIZE = 8
__C.INPUT.DATA_DIR = '/data/yinzhiyu/datasets/167' 
#__C.INPUT.FEATURE = 'features/167'
__C.INPUT.INDEX_DIR = '/home/yzy/workspace/Graph-Memory-Networks/data' 
__C.INPUT.TRAIN_INDEX = 'Index_AID_167_database_train.txt'
__C.INPUT.TEST_INDEX = 'Index_AID_167_database_test.txt'
__C.INPUT.SDF = 'AID_167_database_all.sdf'
__C.INPUT.INDEX = 'Index_AID_167_database_all.txt'

#------Training配置-------#
__C.TRAIN = edict()

__C.TRAIN.BATCH_SIZE = 1
__C.TRAIN.NUM_FRAMES = 64
__C.TRAIN.LEARNING_RATE_BASE = 0.001
__C.TRAIN.DECAY_STEP = 1200
__C.TRAIN.DECAY_RATE = 0.1
__C.TRAIN.DROPOUT_KEEP_PROB = 0.5
__C.TRAIN.REGULARIZATION_SCALE = 0.0001 #??
__C.TRAIN.MAX_ITE = 4000
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.MAX_MODELS_TO_KEEP = 50
__C.TRAIN.SAVED_MODEL_PATTERN = '/data/yinzhiyu/results/Graph-Memory-Networks/models'

#------Valid配置-------#
__C.VALID = edict()

__C.VALID.BATCH_SIZE = 1
__C.VALID.NUM_FRAMES = 64

__C.NETWORK = edict()
__C.NETWORK.STEPS = 10
__C.NETWORK.DIM_A = 10#???
__C.NETWORK.MEM_SIZE = 150
#__C.NETWORK.CELL_SIZE = 1068 #162*(4+2)+(90+6)
__C.NETWORK.CELL_SIZE = 96
__C.NETWORK.ALPH = 0.5#??????
__C.NETWORK.BETA = 0.5#??????
__C.NETWORK.CTRL_STATE_SIZE = 50 #???
__C.NETWORK.BOND_SIZE = 162 #
__C.NETWORK.ONE_HOT_DIM = 90
__C.NETWORK.NUM_BOND_TYPE = 4
__C.NETWORK.MAX_NUM_DEGREE = 6
def get_output_dir(config_file_name):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.EXP_DIR, osp.basename(config_file_name)))
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.iteritems():
    # a must specify keys that are in b
    if not b.has_key(k):
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                         'for config key: {}').format(type(b[k]),
                                                      type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)
