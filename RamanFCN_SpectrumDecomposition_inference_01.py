#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:48:17 2024

@author: nibio
"""






### 
from scipy import interpolate
import random
import os
import os.path
import h5py
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL
import pandas
import scipy.io
import matplotlib
matplotlib.use('Agg')
%matplotlib inline

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy import genfromtxt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import torchvision.utils as utils
###
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as udata
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as utils
import time
import scipy.io as sio
import logging
import datetime
from math import sqrt

import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
#from google.colab import output
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pylab import cm
import torch.optim
import random
#
# from skimage.metrics import peak_signal_noise_ratio
# from scipy.signal import savgol_filter

from torch.nn.parameter import Parameter
#
def get_random_SRF(vector_length, noise_type='u', var=1./10):
    x_shape = (3, vector_length)
    x = torch.zeros(x_shape)
    #
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    x *= var     
    return x.numpy()
#
import numpy as np
import scipy.io as sio
import collections
import scipy.misc

############
def Rand(start, end, num):
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res
##
def l_2_1_loss(im_true, im_fake):
    error = ((((im_fake-im_true).pow(2)).sum(1)).pow(1/2)).sum(0)
    return error
#

########  data preparation !
import rasterio
import torch.utils.data as udata
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

###

import numpy as np
import chardet


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


import matplotlib.pyplot as plt



# create new data by randomly combining all 13 components with 
# different ratios which share non-netative and sum to one property:)

import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def component_mixing(spectra_raw, num, num_in):# num_in, number taken from the library;num_out, number taken from outsiders
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
    ## as the spectra used are raw
    return new_sepctra_in_0, a_in


# load the raw spectra
file_list = glob.glob(os.path.join("EasyCID/Samples/components", "*.txt"))
for f_i in range(len(file_list)):
# for f_i in range(7):
    spectrum_i = parseBWTekFile(file_list[f_i], (file_list[f_i].split("/")[-1]).split(".")[0])
    if f_i ==0:
        spectra_raw = spectrum_i["spectrum"].reshape(1,-1)
    else:
        spectra_raw = np.concatenate((spectra_raw, spectrum_i["spectrum"].reshape(1,-1)), axis = 0)

# normalize the raw spectra to 0-1 range
spectra_raw_n = spectra_raw.copy()
for i in range(spectra_raw.shape[0]):
        spectra_raw_n[i, :] = (spectra_raw[i, :] - np.min(spectra_raw[i, :])) / (np.max(spectra_raw[i, :]) - np.min(spectra_raw[i, :]))



###
import torch
import torch.nn as nn
##
class RamanFCN(nn.Module):
    def __init__(self):
        super(RamanFCN, self).__init__()
        #
        #self.out_channels = out_channels
        self.conv11 = nn.Linear(1976,1524, bias = False)
        self.conv12 = nn.Linear(3500,1500, bias = False)
        self.conv13 = nn.Linear(5000,100, bias = False)
        self.conv14 = nn.Linear(100,13, bias = False)
        #
        self.relu = nn.ReLU()
        #
    def forward(self, x):
        layer_11 = self.conv11(x)
        layer_11 = self.relu(layer_11)
        stack_layer_11 = torch.cat((x,layer_11), dim = 1) #60
        #
        layer_12 = self.conv12(stack_layer_11)
        layer_12 = self.relu(layer_12)
        stack_layer_12 = torch.cat((stack_layer_11, layer_12), dim = 1) #120
        #
        layer_13 = self.conv13(stack_layer_12)
        layer_13 = self.relu(layer_13)
        #
        out = self.conv14(layer_13)
        out = self.relu(out)
        #
        return out
## load the models if you have trained ones 
ramanFCN = RamanFCN().cuda().float()
resume_ramanFCN = os.path.join(os.getcwd(), 'ramanFCN_RamanSpectrumDecomposition_01.pth')
checkpoint_ramanFCN = torch.load(resume_ramanFCN)
ramanFCN.load_state_dict(checkpoint_ramanFCN['state_dict'])

##
num = 1600# total number of spectra
ts_sise = 300

for num_keep in range(1,14):
    print(num_keep)
    
    spectrum_augB_0, ratios_aug_0 = component_mixing(spectra_raw_n, num, num_keep_i)
    spectrum_augB_1, ratios_aug = spectrum_augB_0[-ts_sise:,:], ratios_aug_0[-ts_sise:,:]
    # save the ratios in case you need to check anything
    if num_keep ==1:
        ratios_aug_all = ratios_aug
    else:
        ratios_aug_all = np.concatenate((ratios_aug_all, ratios_aug), axis = 0)
    ## normalize the newly created spectra and baseline
    spectrum_augB = spectrum_augB_1.copy()
    for i in range(spectrum_augB_1.shape[0]):
            spectrum_augB[i, :] = (spectrum_augB_1[i, :] - np.min(spectrum_augB_1[i, :])) / (np.max(spectrum_augB_1[i, :]) - np.min(spectrum_augB_1[i, :]))
    
    ###########################
    #
    label_ratio = torch.tensor(ratios_aug[:,:]).cuda()
    label_ratio = label_ratio.clamp(1e-8, 1.0-1e-8)
    #
    input_spectra = torch.tensor(spectrum_augB[:,:]).cuda()
    input_spectra = input_spectra.clamp(1e-8, 1.0-1e-8)
    ###########################
    with torch.no_grad():
        #
        out_ratio_v = ramanFCN(input_spectra)
        out_ratio_v = out_ratio_v.clamp(1e-8, 1e+8)
        #
        out_ratio_s = out_ratio_v/out_ratio_v.sum(1, keepdim = True)
        out_ratio_s = out_ratio_s.clamp(1e-8, 1.0-1e-8)
        ##
        loss_ratio_i = L1_loss(label_ratio + torch.tensor(1e-8).cuda(), out_ratio_s + torch.tensor(1e-8).cuda())
        #
        mae_ratio_list.append(loss_ratio_i.item())
        ########
        ratio_out_np = out_ratio_s.detach().cpu().squeeze().numpy()
    #
    if num_keep ==1:
        ratios_out_all = ratio_out_np
    else:
        ratios_out_all = np.concatenate((ratios_out_all, ratio_out_np), axis = 0)

####
from numpy import savetxt
from sklearn.metrics import accuracy_score

# calculate the accuracy
dt_thr_ls = [0.00, 0.01, 0.05]
for dtr_i in dt_thr_ls:
    print(dtr_i)
    acc_dtr_i = np.zeros(13)
    for c_i in range(13):
        ratios_gt_ci = ratios_out_all[c_i*300:(c_i+1)*300,:]
        # convert to labels
        label_gt = np.zeros(300,13)
        label_gt[np.where(ratios_gt_ci> dtr_i)] = 1
        #
        ratios_pred_ci = ratios_out_all[c_i*300:(c_i+1)*300,:]
        label_pred = np.zeros(300,13)
        label_pred[np.where(ratios_pred_ci> dtr_i)] = 1
        #
        acc_ci = accuracy_score(label_gt, label_pred)
        print("comp = {}, acc = {} with DRT of {}".format(c_i, acc_ci, dtr_i))
        acc_dtr_i[c_i] = acc_ci

## save the accuracy
# accuracy_out_dtr_i = np.concatenate((np.linspace(1,13,13, endpoint=True).reshape(-1,1), acc_dtr_i.reshape(-1,1)), axis = 1)




