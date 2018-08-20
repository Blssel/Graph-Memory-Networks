#coding:utf-8

import numpy as np
import tensorflow as tf
from train import cfg

class GraphMemNet():
  def __init__(self,cfg):
    self.cfg = cfg

  def _write(self,mem,ctrl_state,adj):
    """
    adj:a tensor of shape [batch_size,]
    """
    mem_old = mem
    mem = tf.reshape(tf.contrib.layers.fully_connected(mem,self.cfg.NETWORK.MEM_SIZE),[self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE, self.cfg.NETWORK.CELL_SIZE]) +\
          tf.reshape(tf.contrib.layers.fully_connected(tf.concat([ctrl_state]*self.cfg.NETWORK.MEM_SIZE,axis=1),self.cfg.NETWORK.MEM_SIZE),[self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE,self.cfg.NETWORK.CELL_SIZE])
    for bond in range(self.cfg.NETWORK.NUM_BOND):
      mem_exp = tf.concat([mem]*self.cfg.NETWORK.MEM_SIZE,axis=1)
      # concat　bond特征
      bond_vec = np.zeros((self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE, self.cfg.NETWORK.MEM_SIZE, self.cfg.NETWORK.BOND_SIZE),dtype=np.float)
      bond_vec[:,:,:,bond] = 1.0
      bond_vec = tf.convert_to_tensor(bond_vec)
      mem_bond = tf.concat(mem_exp,tf.onehot],axis=2)
      # 制作mask
      mask = adj[:,bond,:,:]  #shape=[batch_size,mem_size,mem_size]
      mask = tf.cast(tf.reshape(mask,[self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE, self.cfg.NETWORK.MEM_SIZE,1]),tf.float32)
      # 保留每个cell的邻接cell
      mem_bond = tf.multiply(mem_bond,mask)
      # attention暂时略过？？？？
      # 每个cell的邻接cell求和
      mem_bond_summ = tf.reduce_mean(mem_bond,axis=2) #shape [batch_size,mem_size,bond_size]
      mem += tf.reshape(tf.contrib.layers.fully_connected(mem_bond_summ,self.cfg.NETWORK.MEM_SIZE),[self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE,self.cfg.NETWORK.CELL_SIZE])
    # 更新cell值
    mem = self.cfg.NETWORK.BETA*mem + (1-self.cfg.NETWORK.BETA)*mem_old
    return mem

  def _attention(self,mem,ctrl_state):
    a = tf.reshape(tf.contrib.layers.fully_connected(mem, self.cfg.NETWORK.DIM_A,biases_initializer=None),\
                   [self.cfg.TRAIN.BATCH_SIZE,self.cfg.NETWORK.MEM_SIZE,-1]) + \
        tf.contrib.layers.fully_connected(ctrl_state, self.cfg.NETWORK.DIM_A)
    p = tf.reshape(tf.contrib.layers.fully_connected(a,1),\
                   [self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE])
    p = tf.nn.softmax(p) #shape=[batch_size*mem_size]

  def _read(self,mem,ctrl_state):
    # 加权求和
    p=self._attention(mem)
    p=tf.reshape(p,[self.cfg.TRAIN.BATCH_SIZE, self.cfg.NETWORK.MEM_SIZE,1])
    summary_m=tf.reduce_mean(tf.multiply(mem,p),axis=1) #after multiply,shape=[batch_size*max_nodes*cfg.dim_m].shape=[batch_size*dim_m]
    ctrl_state = tf.contrib.layers.fully_connected(summary_m,self.cfg.NETWORK.CTRL_STATE_SIZE,biases_initializer=None)+\
                  tf.contrib.layers.fully_connected(ctrl_state,self.cfg.NETWORK.CTRL_STATE_SIZE)
    return ctrl_state

  def _queryInput(self,query):
    """
    query:batch_size*cfg.QUERY_SIZE
    """
    return tf.contrib.layers.fully_connected(query, cfg.INPUT.CTRL_STATE_SIZE )  # define and intialize the controller state
  def inference(self, mem, adj, query):
    """
    mem:batch_size*mem_size*cell_size
    adj:batch_size*mem_size*num_bond_type*mem_size
    """
    # 定义controller初始状态
    ctrl_state = tf.constant(0.0,shape=[cfg.TRAIN.BATCH_SIZE,cfg.INPUT.CTRL_STATE_SIZE],dtype=tf.float32

    # 输入query
    ctrl_state = _queryInput(query)

    # 循环读出及写入
    for step in range(self.cfg.NETWORK.STEPS):
      # 读入
      ctrl_state_old = ctrl_state
      ctrl_state = _read(mem,ctrl_state)
      ctrl_state = self.cfg.NETWORK.ALPH*ctrl_state + (1-self.cfg.NETWORK.ALPH)*ctrl_state_old

      # 写出
      mem = _write(mem,ctrl_state,adj) 

    return mem
             
