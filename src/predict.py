#coding:utf-8
import argparse
import pprint
import os
import sys
import time
import tensorflow as tf
import numpy as np
import os.path as osp
import dataset_factory.reader_trainval as reader
sys.path.append(os.getcwd())
from nets.memorynet import GraphMemNet
from config import cfg,cfg_from_file ,get_output_dir
from loss import loss_func

__author__='Zhiyu Yin'

def _parse_args():
  parser=argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg',dest='cfg_file',help='optional config file',default=None,type=str)
  args=parser.parse_args()
  return args

def main():

  #-------------解析参数-------------#
  args=_parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  #读取args.cfg_file文件内容并融合到cfg中
  pprint.pprint(cfg)

  #-------------任务相关配置-------------#
  #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
  #os.environ['CUDA_VISBLE_DEVICES'] = cfg.GPUS
  #os.environ['CUDA_VISBLE_DEVICES'] = '0'
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别

  #-------------搭建计算图-------------#
  with tf.device('/cpu:0'):
    # 操作密集型放在CPU上进行
    query = tf.placeholder(dtype=tf.float32,shape=[None,2])
    num_gpus = len(cfg.GPUS.split(','))
    # 建立dataset，获取iterator
    ite_val = reader.get_dataset_iter_valid(cfg)
    mem_val, mem_adj_val, gt_val = ite_val.get_next()
      
  # 在GPU上运行预测
  with tf.variable_scope(tf.get_variable_scope()) as vscope: # 见https://github.com/tensorflow/tensorflow/issues/6220
    for i in range(num_gpus):
      with tf.device('/gpu:%d'%i), tf.name_scope('GPU_%d'%i) as scope: 
        # 获取网络，并完成前传
        graph_mem_net = GraphMemNet(cfg)
        logits = graph_mem_net.inference(mem_val, mem_adj_val, query)

        tf.get_variable_scope().reuse_variables()
  # saver
  model_variables_map_save={}
  for variable in tf.trainable_variables():
    model_variables_map_save[variable.name.replace(':0', '')] = variable
  print '#####################################################'
  for save_item in model_variables_map_save.keys():
    print save_item
  print '#####################################################'
  saver_save = tf.train.Saver(var_list=model_variables_map_save,max_to_keep=cfg.TRAIN.MAX_MODELS_TO_KEEP)


  #-------------启动Session-------------#
  # (预测验证集，求取精度)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config =tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
  with tf.Session(config = config) as sess:
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    #加载pretrained models
    tf.global_variables_initializer().run()
    saver_save.restore(sess,'/data/yinzhiyu/results/Graph-Memory-Networks/models-1')

    sess.graph.finalize()
    start_time = time.time()
    query_ = np.array([[1.0,0.0]]*cfg.TRAIN.BATCH_SIZE,dtype=np.float)
    num_pred = 0
    T = 0
    P = 0
    TP = 0
    for i in range(cfg.TRAIN.MAX_ITE):
      try:
        print 'predicting %dth mol'%i
        output,gt = sess.run([logits,gt_val],feed_dict={query:query_},options=run_options)
        ind_out = np.argmax(output,axis=1)[0]
        ind_gt = np.argmax(gt,axis=1)[0]
        num_pred += 1
        if ind_gt == 0:
          P+=1
          if ind_out==ind_gt:
            T += 1
            TP += 1
        else:
          if ind_out==ind_gt:
            T += 1
      except:
        pre = float(T)/float(num_pred)
        recall = float(TP)/float(P)
        print 'finished!!!!'
        print 'F1 score is %.2f '%(100.0*2.0*(pre*recall/(pre+recall)))
        break 

if __name__=='__main__':
  main() 
