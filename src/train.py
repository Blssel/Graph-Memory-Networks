#coding:utf-8
import argparse
import pprint
import os
import sys
import time
import tensorflow as tf
import numpy as np
import os.path as osp
import dataset_factory.i3d_data_reader as reader
sys.path.append(os.getcwd())
from nets import i3d
from config import cfg,cfg_from_file ,get_output_dir
from loss import tsn_loss
from utils.view_ckpt import view_ckp
from memory import Controller

def average_gradients(tower_grads):
  average_grads=[]
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g,0)
      grads.append(expanded_g)
    grad = tf.concat(grads,0)
    grad = tf.reduce_mean(grad,0)
    v = grad_and_vars[0][1]
    grads_and_var = (grad, v)
    average_grads.append(grads_and_var)

  return average_grads

def _parse_args():
  parser=argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg',dest='cfg_file',help='optional config file',default=None,type=str)
  args=parser.parse_args()
  return args

def repeat(v,times):
  return np.repeat(v,times,2)
def main():
  #-------------parse arguments-------------#
  args=_parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  #读取args.cfg_file文件内容并融合到cfg中
  pprint.pprint(cfg)

  #-------------some configurations-------------#
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ['CUDA_VISBLE_DEVICES']= cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别

  # -------------build the graph-------------#
  controller = Controller()
  with tf.VariableScope("Query_input"):
    query = tf.one_hot([cfg.TASK_NUM - 1] * cfg.BATCH_SIZE, depth=cfg.TASK_NUM)[0]  # generate corresponding query,shape=[cfg.BATCH_SIZE,cfg.TASK_NUM]
    controller.queryInput(cfg, query)
  with tf.VariableScope("Read_Write"):
    # get m_t
    for i in range(cfg.NUM_ITE):
      with tf.VariableScope("Read_%d"%i):
        controller.read()

        # attention
        a=tf.reshape(tf.contrib.layers.fully_connected(graphmem.mem,cfg.dim_a,biase=None,biases_initializer=None), [cfg.BATCH_SIZE,None])
          +tf.contrib.layers.fully_connected(controller_state,cfg.dim_a)
        p=tf.nn.softmax(tf.reshape(tf.contrib.layers.fully_connected(a,1,)
                                   , [cfg.BATCH_SIZE, graphmem.mem.get_shape().as_lsit()]))
        m=tf

      with tf.VariableScope("Write_%d"%i):





        controller_state=tf.contrib.layers.fully_connected()