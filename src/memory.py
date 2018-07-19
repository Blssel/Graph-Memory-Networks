#coding:utf-8
import numpy as np
import tensorflow as tf

class Cell:
  '''
  args:
    neighbors: a 2-dim list,each line stores the ids of a specific kind of neighbors
  '''
  def __init__(self,feature,neighbors):
    self.feature = feature
    self.neighbors=neighbors

class GraphMemory:
  cells = [] #considering batch processing, the size should be:batch_size*cell_size
  def __init__(self,batch_size):
    self.batch_size = batch_size
    self.construct_mem()  #construct memory

  def construct_mem(self):
    '''Assign id for each node in order, then put them into corresponding memory cell.
    '''
    # instantiate a Cell and append to cells


class Controller:
  """each time, the controller process a batch of data
  """
  def __init__(self):

  def read(self):
    #
  def write(self):

