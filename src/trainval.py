#coding:utf-8
import argparse
import pprint
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISBLE_DEVICES'] = '1'
import sys
import time
import tensorflow as tf
import numpy as np
import os.path as osp
import dataset_factory.reader_trainval as reader
sys.path.append(os.getcwd())
from nets.memorynet import GraphMemNet  #!!!!!
from config import cfg,cfg_from_file ,get_output_dir
from loss import loss_func

__author__='Zhiyu Yin'

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

def main():

  #-------------解析参数-------------#
  args=_parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  #读取args.cfg_file文件内容并融合到cfg中
  pprint.pprint(cfg)

  #-------------任务相关配置-------------#
  #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
  #os.environ['CUDA_VISBLE_DEVICES'] = cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别

  #-------------搭建计算图-------------#
  with tf.device('/cpu:0'):
    # 操作密集型放在CPU上进行
    query = tf.placeholder(dtype=tf.float32,shape=[None,2])
    global_step= tf.get_variable('global_step',[],dtype=None,initializer=tf.constant_initializer(0),trainable=False)
    lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE_BASE, global_step, cfg.TRAIN.DECAY_STEP, cfg.TRAIN.DECAY_RATE, staircase=True) # 学习率
    tf.summary.scalar('learnrate',lr)
    opt = tf.train.MomentumOptimizer(lr,cfg.TRAIN.MOMENTUM)  # 优化函数
    #opt = tf.train.GradientDescentOptimizer(lr)  # 优化函数
    num_gpus = len(cfg.GPUS.split(','))
    # 建立dataset，获取iterator
    ite_train = reader.get_dataset_iter(cfg)
    mem, mem_adj, gt = ite_train.get_next()
      
  # 在GPU上运行训练
  #tower_grads = []
  with tf.variable_scope(tf.get_variable_scope()) as vscope: # 见https://github.com/tensorflow/tensorflow/issues/6220
    for i in range(num_gpus):
      with tf.device('/gpu:%d'%i), tf.name_scope('GPU_%d'%i) as scope: 
        #query = np.array([[1.0,0.0]]*cfg.TRAIN.BATCH_SIZE,dtype=np.float)
        #query = tf.cast(tf.convert_to_tensor(query),tf.float32)
        # 获取网络，并完成前传
        #with tf.Graph().as_default():
        graph_mem_net = GraphMemNet(cfg)
        logits = graph_mem_net.inference(mem, mem_adj, query)

        tf.get_variable_scope().reuse_variables()
        # 做一个batch准确度的预测
        prediction = tf.nn.softmax(logits)
        acc_batch = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(gt,1)),tf.float32))
        tf.summary.scalar('acc_on_batch',acc_batch)
        # 求loss
        for variable in tf.global_variables():
          if variable.name.find('weights')>0: # 把参数w加入集合tf.GraphKeys.WEIGHTS，方便做正则化(此句必须放在正则化之前)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS,variable)
        loss = loss_func(cfg,logits, gt, regularization= True)
        tf.summary.scalar('loss',loss)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = opt.minimize(loss, global_step=global_step,var_list=tf.trainable_variables())
   
   
  merged = tf.summary.merge_all()
 

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
    joint_writer = tf.summary.FileWriter(cfg.SUMMARY_DIR, sess.graph)
    summary_writer = tf.summary.FileWriter(cfg.SUMMARY_DIR, sess.graph)

    #初始化变量(或加载pretrained models)
    tf.global_variables_initializer().run()
    #saver_save.restore(sess,'/data/yinzhiyu/results/Graph-Memory-Networks/models-4001')

    sess.graph.finalize()
    start_time = time.time()
    query_ = np.array([[1.0,0.0]]*cfg.TRAIN.BATCH_SIZE,dtype=np.float)
    #query = tf.cast(tf.convert_to_tensor(query),tf.float32)
    for i in range(cfg.TRAIN.MAX_ITE):
      _,learnrate, loss_value, step, summary = sess.run([train_step, lr, loss, global_step,merged],feed_dict={query:query_},options=run_options, run_metadata=run_metadata)
      if i==0:
        start_time = time.time()
      if i % 10 == 0:
        if i>=1:
          end_time = time.time()
          avg_time = (end_time-start_time)/float(i+1)
          print("Average time consumed per step is %0.2f secs." % avg_time)
        print("After %d training step(s), learning rate is %g, loss on training batch is %g." % (step, learnrate, loss_value))

      # 每个epoch验证一次，保存模型
      if i % 2000 == 0:
        print '#############################################'
        print 'saving model...'
        saver_save.save(sess, cfg.TRAIN.SAVED_MODEL_PATTERN, global_step=global_step)
        print 'successfully saved !'
        print '#############################################'
        
      if i % 200 == 0: 
        joint_writer.add_run_metadata(run_metadata, 'step%03d'%i)
        summary_writer.add_summary(summary,i)
      end_time = time.time()
      #print '%dth time step,consuming %f secs'%(i, start_time-end_time)

  summary_writer.close()

if __name__=='__main__':
  main() 
