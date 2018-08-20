#coding:utf-8
import argparse
import pprint
import os
import os.path as osp
import numpy as np
import rdkit
from rdkit import Chem
from config import cfg,cfg_from_file ,get_output_dir

def _parse_args():
  parser=argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg',dest='cfg_file',help='optional config file',default=None,type=str)
  args=parser.parse_args()
  return args


def main():
  args=_parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  #读取args.cfg_file文件内容并融合到cfg中
  pprint.pprint(cfg)
  
  #　读取索引文件
  with open(osp.join(cfg.INPUT.DATA_DIR,cfg.INPUT.INDEX),'r') as f:
    index = f.readlines()

  # 原子数最多的分子共含146个原子,可认为最大为150个，故mem_size=150，167数据集共86个原子，故one-hot设为90维
  suppl = Chem.SDMolSupplier(osp.join(cfg.INPUT.DATA_DIR,cfg.INPUT.SDF))
  for i in range(len(suppl)):
    m=suppl[i]
    if m is None: 
      continue
    # atom的one-hot编码,degree,H attached
    feature_mol=[]
    for atom in m.GetAtoms():
      feature_atom=[]
      one_hot = np.zeros(cfg.NETWORK.ONE_HOT_DIM,dtype=np.float)
      one_hot[atom.GetAtomicNum()]=1.0
      feature_atom.append(one_hot)
      degree = np.array([x.GetAtomicNum() for x in atom.GetNeighbors()]) ######one-hot????
      feature_atom.append(degree)
    for bond in m.GetBonds():
      # degree
      bond_type=np.zeros(cfg.NETWORK.NUM_BOND_TYPE,np.float)
      if bond.GetBondType()==rdkit.Chem.rdchem.BondType.SINGLE:
        bond_type[0]=1.0
      elif bond.GetBondType()==rdkit.Chem.rdchem.BondType.DOUBLE:
        bond_type[1]=1.0
      elif bond.GetBondType()==rdkit.Chem.rdchem.BondType.TRIPLE:
        bond_type[2]=1.0
      else:
        bond_type[3]=1.0
      feature_atom.append(bond_type)
      # is bond in ring
      if bond.IsInRing():
        feature_atom.append(np.array([1.0],dtype=float))
      else:
        feature_atom.append(np.array([0.0],dtype=float))
    # 整理feature_atom，并加入mem,
    feature_atom = np.concatenate(feature_atom)
    feature_mol.append(feature_atom) #此处未padding!!!!!!!!!!!
    feature_mol=np.array(feature_mol)
   
    # 构建邻接矩阵 
    adj=np.zeros((cfg.NETWORK.NUM_BOND_TYPE,cfg.NETWORK.MEM_SIZE,cfg.NETWORK.MEM_SIZE),dtype=np.float)
    for ii in range(len(m.GetAtoms())):
      for jj in range(len(m.GetAtoms())):
        if m.GetBondBetweenAtoms(ii,jj)==None:
          continue
        else:
          if m.GetBondBetweenAtoms(ii,jj).GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            adj[0][ii][jj]=1.0
          elif m.GetBondBetweenAtoms(ii,jj).GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
            adj[1][ii][jj]=1.0
          elif m.GetBondBetweenAtoms(ii,jj).GetBondType() == rdkit.Chem.rdchem.BondType.TRIPLE:
            adj[2][ii][jj]=1.0
          else:
            adj[3][ii][jj]=1.0

    #　保存
    file_name = index[i].split()[0]
    mol_fea_adj = np.array([feature_mol, adj,
                            np.array([1.0,0.0],dtype=np.float) if index[i].split()[1]=='Active' else np.array([0.0,1.0],dtype=np.float)])
    np.save(osp.join(cfg.INPUT.DATA_DIR, cfg.INPUT.FEATURE, file_name+'.npy'),mol_fea_adj) 
    print '%d precessed\n'%i

      
  """
  for atom in m.GetAtoms():
  print(atom.GetAtomicNum())
  print m.GetAtomWithIdx(2).GetSymbol()
  """
if __name__=='__main__':
  main()
