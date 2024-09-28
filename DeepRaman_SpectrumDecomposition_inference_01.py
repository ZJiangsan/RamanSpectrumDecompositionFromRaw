#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:20:38 2024

@author: nibio
"""




import os
import glob
import tensorflow as tf
import numpy as np
from sklearn.linear_model import enet_path
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from scipy.sparse import spdiags,eye,csc_matrix, diags
import copy
import csv
from DeepRaman.DeepRaman.testing_Multi_component_mixture import SpatialPyramidPooling, WhittakerSmooth_MAT, airPLS_MAT
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import chardet
from sklearn.metrics import accuracy_score


def locate_line(lines, keyword):
    for i in range(0, len(lines)):
        fr = lines[i].find(keyword)
        if fr != -1:
            return i
    return "CAN'T FIND";


def parseBWTekFile(filepath, filename, select_ramanshift=False, xname='Raman Shift', yname="Dark Subtracted #1"):
    file_content = open(filepath).read()
    fc_splits = file_content.split('\n')
    spectrum = {}
    spectrum['name'] = filename[0:(len(filename) - 4)]
    spectrum['excition'] = float(fc_splits[locate_line(fc_splits, 'laser_wavelength')].split(';')[1])
    spectrum['integral_time'] = float(fc_splits[locate_line(fc_splits, 'intigration times')].split(';')[1])

    nTitle = locate_line(fc_splits, xname)

    Xunit = fc_splits[nTitle].split(';')
    AXI = Xunit.index(xname)
    DSI = Xunit.index(yname)
    xaxis_min = nTitle + 1 + int(fc_splits[locate_line(fc_splits, 'xaxis_min')].split(';')[1])
    xaxis_max = nTitle + 1 + int(fc_splits[locate_line(fc_splits, 'xaxis_max')].split(';')[1])

    nLen = xaxis_max - xaxis_min
    spectrum["axis"] = np.zeros((nLen,), dtype=np.float64)
    spectrum["spectrum"] = np.zeros((nLen,), dtype=np.float64)
    for i in range(xaxis_min, xaxis_max):
        fc_ss = fc_splits[i].split(';')
        spectrum["axis"][i - xaxis_min] = float(fc_ss[AXI])
        spectrum["spectrum"][i - xaxis_min] = float(fc_ss[DSI])
    if select_ramanshift:
        inds = np.logical_and(spectrum["axis"] > 160, spectrum["axis"] < 3000)
        spectrum["axis"] = spectrum["axis"][inds]
        spectrum["spectrum"] = spectrum["spectrum"][inds].astype(np.float64)
    return spectrum


def component_mixing(spectra_raw, num, num_in):
    spectra_raw_in = spectra_raw[:,:]
    #
    a_in = np.zeros((num, spectra_raw_in.shape[0]))
    #
    for i in range(num):
        ratios_in_0 = np.random.rand(spectra_raw.shape[0])
        # randomly keep the ratios of num_in components
        idx_in = random.sample(list(range(spectra_raw_in.shape[0])), spectra_raw_in.shape[0] - num_in)
        ratios_in_0[idx_in] = 0
        #
        ratios_in = np.array(ratios_in_0) / sum(ratios_in_0)
        #
        a_in[i, :] = ratios_in
    #
    print("a_in {}".format(np.sum(a_in, axis = 1)))
    #
    new_sepctra_in_0 = np.dot(a_in, spectra_raw_in) 
    return new_sepctra_in_0, a_in

##
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_list):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_list = pool_list

    def forward(self, x):
        num_samples, num_channels, height, width = x.size()
        output = []

        for pool_size in self.pool_list:
            pool = F.adaptive_max_pool2d(x, output_size=(pool_size, pool_size))
            output.append(pool.view(num_samples, -1))

        return torch.cat(output, 1)

class DeepRamanModel(nn.Module):
    def __init__(self):
        super(DeepRamanModel, self).__init__()
        self.convA1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bnA1 = nn.BatchNorm1d(32)
        self.poolA1 = nn.MaxPool1d(3, stride=3)

        self.convB1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bnB1 = nn.BatchNorm1d(32)
        self.poolB1 = nn.MaxPool1d(3, stride=3)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.spp = SpatialPyramidPooling([1, 2, 3, 4])

        self.fc1 = nn.Linear(32 * (1 + 4 + 9 + 16), 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1)

        self._initialize_weights()

    def forward(self, x):
        inputA = x[:, 0, :].unsqueeze(1)
        inputB = x[:, 1, :].unsqueeze(1)

        xA = F.relu(self.bnA1(self.convA1(inputA)))
        xA = self.poolA1(xA)

        xB = F.relu(self.bnB1(self.convB1(inputB)))
        xB = self.poolB1(xB)

        x = torch.cat((xA, xB), dim=2)
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.spp(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.sigmoid(self.fc2(x)).squeeze()

        return x
    


# 
deepRaman = DeepRamanModel().cuda()
criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(deepRaman.parameters(), lr=0.01)  # Reduced learning rate
#
## load the models if you have trained one
resume_deepRaman = os.path.join(os.getcwd(), 'DeepRaman_RamanSpectrumDecomposition_01.pth') # it is the wrong name only
checkpoint_deepRaman = torch.load(resume_deepRaman)
deepRaman.load_state_dict(checkpoint_deepRaman['state_dict'])

## pure components
file_list = glob.glob(os.path.join("EasyCID/Samples/components", "*.txt"))
#
DBcoms = range(1,14)
print('Database：',DBcoms)
#
for f_i in range(len(file_list)):
    spectrum_i = parseBWTekFile(file_list[f_i], (file_list[f_i].split("/")[-1]).split(".")[0])
    if f_i ==0:
        spectra_raw = spectrum_i["spectrum"].reshape(1,-1)
    else:
        spectra_raw = np.concatenate((spectra_raw, spectrum_i["spectrum"].reshape(1,-1)), axis = 0)


spectra_raw.shape
spectrum_pure_sc = spectra_raw.copy()
for i in range(spectra_raw.shape[0]):
        spectrum_pure_sc[i, :] = spectra_raw[i, :] / np.max(spectra_raw[i, :])

#


num = 1600 # how many are you going to augment
ts_sise = 300
# num_keep = 13
accuracy_total = []
ratio_mae_all = []
#
num_pred_list_full = []

for num_keep in range(1,14):
    # num_keep = 1
    print(num_keep)
    
    spectrum_augB_00, ratios_aug0 = component_mixing(spectrum_pure_sc, num, num_keep)
    spectrum_augB_0, ratios_aug = spectrum_augB_00[-ts_sise:,:], ratios_aug0[-ts_sise:,:]
    
    ## normalize the newly created spectra and baseline
    spectrum_mix_sc = spectrum_augB_0.copy() # 
    for i in range(spectrum_augB_0.shape[0]):
            spectrum_mix_sc[i, :] = spectrum_augB_0[i, :] /np.max(spectrum_augB_0[i, :])
    
    X = np.zeros((spectrum_mix_sc.shape[0]*spectrum_pure_sc.shape[0],2,1976,1))
    
    for p in range(spectrum_mix_sc.shape[0]):
        for q in range(spectrum_pure_sc.shape[0]):
            X[int(p*spectrum_pure_sc.shape[0]+q),0,:,0] = spectrum_mix_sc[p,:]
            X[int(p*spectrum_pure_sc.shape[0]+q),1,:,0] = spectrum_pure_sc[q,:]
    
    
    _custom_objects = {
       "SpatialPyramidPooling" :  SpatialPyramidPooling,
       }  
    
    y = deepRaman.predict(X)
    
    ###  for ratio estimation, these pure and mixture spectra have to be preprocessed
    spectrum_pure_scS = WhittakerSmooth_MAT(spectrum_pure_sc, lamb=1) # smoothing
    spectrum_pure_scSB = airPLS_MAT(spectrum_pure_scS, lamb=10, itermax=10) # baseline correction
    spectrum_mix_scBS = WhittakerSmooth_MAT(spectrum_mix_sc, lamb=1)
    spectrum_mix_scBSB = airPLS_MAT(spectrum_mix_scBS, lamb=10, itermax=10)
    
    ##  ratio estimation
    predic_ratios = np.zeros((spectrum_mix_scBSB.shape[0], 13))
    predic_comp = np.zeros((spectrum_mix_scBSB.shape[0], 13))
    gt_comp = np.zeros((spectrum_mix_scBSB.shape[0], 13))
    for cc in range(spectrum_mix_scBSB.shape[0]):
        # get the components based on GT ratio
        com_gt_ct_0 = ratios_aug.copy()[cc,:]
        com_gt_ct = com_gt_ct_0[np.where(com_gt_ct_0!=0)]
        com_gt_0 = ratios_aug.copy()[cc,:]
        thre_num = 0.00
        com_gt_0[com_gt_0 <= thre_num] = 0
        com_gt_0[com_gt_0 > thre_num] = 1
        com_gt = np.where(com_gt_0==1)[0]
        gt_comp[cc,:] = com_gt_0
        print("com_gt based on ratio {}".format(int(np.sum(com_gt_0))))
        print("com_gt based on num_keep {}".format(num_keep))
        print("Ground truth of components in the spectrum {}".format(com_gt))
        print("Ground truth ratios {}".format(com_gt_ct))
        #####
        com=[]
        coms = []
        ra2 = []
        for ss in range(cc*spectrum_pure_scSB.shape[0],(cc+1)*spectrum_pure_scSB.shape[0]):
            if y[ss,1]>=0.5:
                com.append(ss%spectrum_pure_scSB.shape[0])
        print("com_pred content {}".format(com)) 
        print("com_pred number {}".format(len(com)))
        predic_comp[cc,com] = 1
        num_pred_list_full.append(len(com))
        #
        if len(com) ==0:
            print("Did not detect any components")
            com = [0]
        X_n = spectrum_pure_scSB[com]
        coms = [DBcoms[com[h]] for h in range(len(com))]
    
        _, coefs_lasso, _ = enet_path(X_n.T, spectrum_mix_scBSB[cc,:], l1_ratio=0.96,
                                  positive=True)
        
        ratio = coefs_lasso[:, -1]
        ratio_sc = copy.deepcopy(ratio)
    
        for ss2 in range(ratio.shape[0]):
            ratio_sc[ss2]=ratio[ss2]/np.sum(ratio)
    
        print("cc = {}".format(cc))
        print('The',cc, 'spectra may contain {} components including:'.format(len(coms)),coms)
        print('The corresponding ratio is:', ratio_sc)
        ##
        # assgin predicted ratio to the right place
        predic_ratios[cc,com] = ratio_sc
    ##
    accuracy_i = accuracy_score(predic_comp, gt_comp)
    print("{}, accuracy: {}".format(num_keep, accuracy_i))
    accuracy_total.append(accuracy_i)
    ratio_mae_i = np.mean(np.abs(ratios_aug - predic_ratios))
    print("{}, ratio_mae_i: {}".format(num_keep, ratio_mae_i))
    ratio_mae_all.append(ratio_mae_i)


