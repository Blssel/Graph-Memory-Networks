import os
import random

with open('Index_AID_167_database_all.txt') as f:
  lines = f.readlines()

train_file = 'Index_AID_167_database_all.txt'.replace('all','train')
test_file = 'Index_AID_167_database_all.txt'.replace('all','test')

random.shuffle(lines)
lines_train = lines[0:int(0.8*len(lines))]
lines_test = lines[int(0.8*len(lines))+1:]

if not os.path.exists(train_file):
  with open(train_file,'a') as f:
    for i in range(len(lines_train)):
      if i!=len(lines_train)-1:
        f.write(lines_train[i].strip()+'\n')
      else:
        f.write(lines_train[i].strip())
if not os.path.exists(test_file):
  with open(test_file,'a') as f:
    for i in range(len(lines_test)):
      if i!=len(lines_test)-1:
        f.write(lines_test[i].strip()+'\n')
      else:
        f.write(lines_test[i].strip())
