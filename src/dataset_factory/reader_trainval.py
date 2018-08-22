#coding:utf-8
import glob
import os
import os.path as osp
import random
import numpy as np
import cv2
import time
import tensorflow as tf

global cfg
global is_training
cfg, is_training = None, None

def _generator():
  global cfg
  with open(osp.join(cfg.INPUT.INDEX_DIR,cfg.INPUT.TRAIN_INDEX)) as f:
    mol_list = f.readlines()
    print len(mol_list)
  for i in range(len(mol_list)):
    mol_id = mol_list[i].split()[0]
    mol_feature_path = osp.join(cfg.INPUT.DATA_DIR, mol_id+'.npy')
    if not os.path.exists(mol_feature_path):
      continue
    mol_adj_gt = np.load(mol_feature_path)
    mol_feature = mol_adj_gt[0]
    mol_adj = mol_adj_gt[1]
    mol_gt = mol_adj_gt[2]
    mol_feature=mol_feature[:,0:96]
    yield (mol_feature, mol_adj, mol_gt)


def get_dataset_iter(config):
  """
  读取数据，预处理，组成batch，返回
  """
  global cfg
  cfg = config

  #dataset_train=tf.data.Dataset.from_generator(_generator,(tf.float32, tf.float32, tf.float32),
  #                                                        (tf.TensorShape([cfg.NETWORK.MEM_SIZE,cfg.NETWORK.CELL_SIZE]),tf.TensorShape([cfg.NETWORK.NUM_BOND_TYPE,cfg.NETWORK.MEM_SIZE,cfg.NETWORK.MEM_SIZE]),tf.TensorShape([2]) ))
  dataset_train=tf.data.Dataset.from_generator(_generator,(tf.float32, tf.float32, tf.float32),
                                                          (tf.TensorShape([150,96]),tf.TensorShape([4,150,150]),tf.TensorShape([2]) ))
  dataset_train = dataset_train.repeat().shuffle(buffer_size=cfg.TRAIN.BATCH_SIZE*20).batch(cfg.TRAIN.BATCH_SIZE).prefetch(buffer_size=10)
  iter_train = dataset_train.make_one_shot_iterator()
  return iter_train

def _generator_valid():
  global cfg
  with open(osp.join(cfg.INPUT.INDEX_DIR,cfg.INPUT.TEST_INDEX)) as f:
    mol_list = f.readlines()
    print len(mol_list)
  for i in range(len(mol_list)):
    mol_id = mol_list[i].split()[0]
    mol_feature_path = osp.join(cfg.INPUT.DATA_DIR, mol_id+'.npy')
    if not os.path.exists(mol_feature_path):
      continue
    mol_adj_gt = np.load(mol_feature_path)
    mol_feature = mol_adj_gt[0]
    mol_adj = mol_adj_gt[1]
    mol_gt = mol_adj_gt[2]
    mol_feature=mol_feature[:,0:96]
    yield (mol_feature, mol_adj, mol_gt)


def get_dataset_iter_valid(config):
  global cfg
  cfg = config

  dataset_valid=tf.data.Dataset.from_generator(_generator_valid,(tf.float32, tf.float32, tf.float32),
                                                          (tf.TensorShape([150,96]),tf.TensorShape([4,150,150]),tf.TensorShape([2]) ) ) 
  dataset_valid = dataset_valid.batch(cfg.VALID.BATCH_SIZE)
  iter_valid = dataset_valid.make_one_shot_iterator()
  return iter_valid
