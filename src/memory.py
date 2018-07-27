#coding:utf-8

import numpy as np
import tensorflow as tf
from train import cfg
'''
class Cell:
  def __init__(self,feature,neighbors):
    self.feature = tf.constant(0,dtype=tf.float32,shape=[cfgs.SHAPE])
    self.neighbors=
'''
class GraphMemory:
  cells = [] #considering batch processing, the size should be:batch_size*cell_size
  '''Initialize an empty graph(or graph memory)
  '''
  mem=tf.placeholder(0,shape=[cfg.batch_size,None,cfg.dim_m])  #batch_size* max_nodes* dim_m
  adj_matrix=None
  relation=[]
  def __init__(self,cfg,max_nodes):
    self.max_nodes = max_nodes
    self.construct_mem()  #construct memory
    self.mem=[[] for i in range(cfg.batch_size)]
  '''
  def construct_mem(self):
    #Assign id for each node in order, then put them into corresponding memory cell.
    # instantiate a Cell and append to cells
  '''


class Controller:
  '''each time, the controller process a batch of data
  state:shape=[batch_size,]
  '''
  def __init__(self):
    self.state = []  # state is a 2-dim tuple


  def queryInput(self,cfg,query):
    self.state.append(
      tf.contrib.layers.fully_connected(query, cfg.DIM_STATE))  # define and intialize the controller state
    self.state.append(None)

  def __attention(self,memory):
    a = tf.reshape(tf.contrib.layers.fully_connected(memory.mem, cfg.dim_a,biases_initializer=None),\
                   [cfg.BATCH_SIZE,memory.max_nodes,None]) + \
      tf.contrib.layers.fully_connected(self.state, cfg.dim_a)
    p = tf.reshape(tf.contrib.layers.fully_connected(a,1),\
                   [cfg.BATCH_SIZE, memory.max_nodes])
    p = tf.nn.softmax(p) #shape=[batch_size*max_nodes]
  def read(self,memory):
    p=self.__attention(memory)
    p=tf.reshape(p,[cfg.BATCH_SIZE,memory.max_nodes,1])
    summary_m=tf.reduce_mean(tf.multiply(memory,p),axis=1) #after multiply,shape=[batch_size*max_nodes*cfg.dim_m].shape=[batch_size*dim_m]
    self.state[1]=tf.contrib.layers.fully_connected(summary_m,cfg.dim_state,biases_initializer=None)+\
                  tf.contrib.layers.fully_connected(self.state[0],cfg.dim_state)
    self.state[0]=self.state[1]


  def __adj_add(self,previous_mem,adj_matrix,relation,r,max_nodes):
    adj_added=np.zeros((cfg.BATCH_SIZE,max_nodes,cfg.dim_m+cfg.dim_link))
    for i in range(cfg.BATCH_SZIE):
      for j in range(max_nodes):
        for k in range(max_nodes):
          if k==r:
            continue
          if adj_matrix[j][k]==r:
             adj_added[i][j]=np.concatenate(previous_mem[i][k], relation[r])
    return adj_added


  def write(self,memory):
    part_a=0
    tmp_state=self.state
    previous_mem = memory.mem
    for rela in memory.relation():
      relt_add_mem=previous_mem
      # 对每个cell，将满足该关系的节点cell与对应relation concate，然后加起来存到此cell的位置，无此关系的cell置零。（此处暂时不用attention）
      adj_added=tf.py_func(__adj_add,[previous_mem, memory.adj_matrix,memory.relation,rela, memory.max_nodes],dtype=[tf.float32,tf.float32,tf.int8,tf.int8,tf.int8])
      part_a+=tf.contrib.layers.fully_connected(adj_added,cfg.dim_m)

    memory.mem=part_a







