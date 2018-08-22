#coding:utf-8

import numpy as np
import tensorflow as tf

def _write(cfg,mem,ctrl_state,adj):
  """
  adj:a tensor of shape [batch_size,]
  """
  with tf.variable_scope('write'):
    mem_old = mem
    mem = tf.reshape(tf.contrib.layers.fully_connected(mem,cfg.NETWORK.CELL_SIZE),[-1, cfg.NETWORK.MEM_SIZE, cfg.NETWORK.CELL_SIZE]) +\
          tf.reshape(tf.contrib.layers.fully_connected(tf.reshape(tf.concat([ctrl_state]*cfg.NETWORK.MEM_SIZE,axis=1),[-1,cfg.NETWORK.MEM_SIZE,cfg.NETWORK.CTRL_STATE_SIZE]),cfg.NETWORK.CELL_SIZE),[-1, cfg.NETWORK.MEM_SIZE,cfg.NETWORK.CELL_SIZE])
    for bond in range(cfg.NETWORK.NUM_BOND_TYPE):
      with tf.variable_scope('bond_type'):
        mem_exp = tf.reshape(tf.concat([mem]*cfg.NETWORK.MEM_SIZE,axis=1),[-1, cfg.NETWORK.MEM_SIZE,cfg.NETWORK.MEM_SIZE,cfg.NETWORK.CELL_SIZE])
        # concat　bond特征
        bond_vec = np.zeros((cfg.TRAIN.BATCH_SIZE, cfg.NETWORK.MEM_SIZE, cfg.NETWORK.MEM_SIZE, cfg.NETWORK.BOND_SIZE),dtype=np.float)
        bond_vec[:,:,:,bond] = 1.0
        bond_vec = tf.cast(tf.convert_to_tensor(bond_vec),tf.float32)
        mem_bond = tf.concat([mem_exp,bond_vec],axis=3)
        # 制作mask
        mask = adj[:,bond,:,:]  #shape=[batch_size,mem_size,mem_size]
        mask = tf.cast(tf.reshape(mask,[-1, cfg.NETWORK.MEM_SIZE, cfg.NETWORK.MEM_SIZE,1]),tf.float32)
        # 保留每个cell的邻接cell
        mem_bond = tf.multiply(mem_bond,mask)
        # attention暂时略过？？？？
        # 每个cell的邻接cell求和
        mem_bond_summ = tf.reduce_mean(mem_bond,axis=2) #shape [batch_size,mem_size,bond_size]
        mem += tf.reshape(tf.contrib.layers.fully_connected(mem_bond_summ,cfg.NETWORK.CELL_SIZE),[-1, cfg.NETWORK.MEM_SIZE,cfg.NETWORK.CELL_SIZE])
        tf.get_variable_scope().reuse_variables()
    tf.get_variable_scope().reuse_variables()
  # 更新cell值
  mem = cfg.NETWORK.BETA*mem + (1-cfg.NETWORK.BETA)*mem_old
  return mem

def _attention(cfg,mem,ctrl_state):
  print '###########'
  print mem.dtype
  print ctrl_state.dtype
  print '###########'
  with tf.variable_scope('attention'):
    a = tf.reshape(tf.contrib.layers.fully_connected(mem, cfg.NETWORK.DIM_A,biases_initializer=None),\
                   [-1,cfg.NETWORK.MEM_SIZE,cfg.NETWORK.DIM_A]) + \
        tf.reshape(tf.contrib.layers.fully_connected(ctrl_state, cfg.NETWORK.DIM_A),\
                   [-1,1,cfg.NETWORK.DIM_A])
    p = tf.reshape(tf.contrib.layers.fully_connected(a,1),\
                   [-1, cfg.NETWORK.MEM_SIZE])
    p = tf.nn.softmax(p) #shape=[batch_size*mem_size]
    tf.get_variable_scope().reuse_variables()
  return p

def _read(cfg,mem,ctrl_state):
  # 加权求和
  with tf.variable_scope('read'):
    p=_attention(cfg,mem,ctrl_state)
    p=tf.reshape(p,[-1, cfg.NETWORK.MEM_SIZE,1])
    summary_m=tf.reduce_mean(tf.multiply(mem,p),axis=1) #after multiply,shape=[batch_size*max_nodes*cfg.dim_m].shape=[batch_size*dim_m]
    ctrl_state = tf.contrib.layers.fully_connected(summary_m,cfg.NETWORK.CTRL_STATE_SIZE,biases_initializer=None)+\
                tf.contrib.layers.fully_connected(ctrl_state,cfg.NETWORK.CTRL_STATE_SIZE)
    tf.get_variable_scope().reuse_variables()
  return ctrl_state

def _queryInput(cfg,query):
  """
  query:batch_size*cfg.QUERY_SIZE
  """
  with tf.variable_scope('query_input'):
    ctrl_state = tf.contrib.layers.fully_connected(query, cfg.NETWORK.CTRL_STATE_SIZE )  # define and intialize the controller state
    tf.get_variable_scope().reuse_variables()
  return ctrl_state

class GraphMemNet():
  def __init__(self,cfg):
    self.cfg = cfg

  def inference(self, mem, adj, query):
    """
    mem:batch_size*mem_size*cell_size
    adj:batch_size*mem_size*num_bond_type*mem_size
    """
    # 定义controller初始状态
    ctrl_state = tf.constant(0.0,shape=[self.cfg.TRAIN.BATCH_SIZE,self.cfg.NETWORK.CTRL_STATE_SIZE],dtype=tf.float32)

    # 输入query
    ctrl_state = _queryInput(self.cfg,query)

    # 循环读出及写入
    #for step in range(self.cfg.NETWORK.STEPS):
    for step in range(8):
      with tf.variable_scope('GraphMemNet',reuse=tf.AUTO_REUSE) as scope_read_wrt:
        # 读入
        ctrl_state_old = ctrl_state
        ctrl_state = _read(self.cfg,mem,ctrl_state)
        ctrl_state = self.cfg.NETWORK.ALPH*ctrl_state + (1-self.cfg.NETWORK.ALPH)*ctrl_state_old

        # 写出
        mem = _write(self.cfg,mem,ctrl_state,adj) 
        scope_read_wrt.reuse_variables()

    #with tf.variable_scope('read',reuse=True) as scope_output:
    # 最后一次读出
    with tf.variable_scope('GraphMemNet',reuse=True) as scope_read_wrt:
      ctrl_state = _read(self.cfg,mem,ctrl_state)
      

    with tf.variable_scope('output') as scope_output:
      return tf.contrib.layers.fully_connected(ctrl_state,2)
             
