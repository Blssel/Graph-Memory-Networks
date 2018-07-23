#coding:utf-8

import numpy as np
import tensorflow as tf
from train import cfg

class Cell:
  '''
  args:
    feature: a tensor of shape(dim_m)
    neighbors: a 2-dim list,each line stores the ids of a specific kind of neighbors
  '''
  def __init__(self,feature,neighbors):
    '''initialize an empty
    :param feature:
    :param neighbors:
    '''
    self.feature = tf.constant(0,dtype=tf.float32,shape=[cfgs.SHAPE])
    self.neighbors=

class GraphMemory:
  cells = [] #considering batch processing, the size should be:batch_size*cell_size
  '''Initialize an empty graph(or graph memory)
  '''
  mem=tf.placeholder(0,shape=[cfg.batch_size,None,cfg.dim_m])  #batch_size* max_nodes* dim_m
  adj_matrix=None
  def __init__(self,cfg):
    self.construct_mem()  #construct memory
    self.mem=[[] for i in range(cfg.batch_size)]
  '''
  def construct_mem(self):
    #Assign id for each node in order, then put them into corresponding memory cell.
    # instantiate a Cell and append to cells
  '''

class Controller:
  '''each time, the controller process a batch of data
  controller_state:shape=[batch_size,]
  '''
  def __init__(self):
    self.controller_state = []  # controller_state is a 2-dim tuple

  def queryInput(self,cfg,query):
    self.controller_state.append(
      tf.contrib.layers.fully_connected(query, cfg.DIM_STATE))  # define and intialize the controller state
    self.controller_state.append(None)

  def __attention(self,memory):
    a = tf.reshape(tf.contrib.layers.fully_connected(memory.mem, cfg.dim_a, biase=None, biases_initializer=None), [cfg.BATCH_SIZE, None]) + \
        tf.contrib.layers.fully_connected(self.controller_state, cfg.dim_a) # a shape=[batch_size*]
    p = tf.nn.softmax(tf.reshape(tf.contrib.layers.fully_connected(a,1,)
                                 ,[cfg.BATCH_SIZE, graphmem.mem.get_shape().as_lsit()]))
    m = tf.
  def read(self,memory):
    # attention


  def write(self):

