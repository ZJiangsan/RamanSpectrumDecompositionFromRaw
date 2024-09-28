#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:39:49 2024

@author: nibio
"""




import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import CubicSpline
from scipy.sparse import spdiags,eye,csc_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import enet_path
from scipy.sparse.linalg import spsolve
from num2words import num2words
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors

def WhittakerSmooth(x, lamb, w):
    m=w.shape[0]
    W=spdiags(w,0,m,m)
    D=eye(m-1,m,1)-eye(m-1,m)
    return spsolve((W+lamb*D.transpose()*D),w*x)

def airPLS(x, lamb=10, itermax=10):
    m=x.shape[0]
    w=np.ones(m)
    for i in range(itermax):
        z=WhittakerSmooth(x,lamb,w)
        d=x-z
        if sum(abs(d[d<0]))<0.001*sum(abs(x)):
            break;
        w[d<0]=np.exp(i*d[d<0]/sum(d[d<0]))
        w[d>=0]=0
    return z

def airPLS_MAT(X, lamb=10, itermax=10):
    B=X.copy()
    for i in range(X.shape[0]):
        B[i,]=airPLS(X[i,],lamb,itermax)
    return X-B

def WhittakerSmooth_MAT(X, lamb=1):
    C=X.copy()
    w=np.ones(X.shape[1])
    for i in range(X.shape[0]):
        C[i,]=WhittakerSmooth(X[i,:], lamb, w)
    return C

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
    ## as the spectra used are raw
    return new_sepctra_in_0, a_in

##  
file_list = glob.glob(os.path.join("EasyCID/Samples/components", "*.txt"))
for f_i in range(len(file_list)):
    spectrum_i = parseBWTekFile(file_list[f_i], (file_list[f_i].split("/")[-1]).split(".")[0])
    if f_i ==0:
        spectra_raw = spectrum_i["spectrum"].reshape(1,-1)
    else:
        spectra_raw = np.concatenate((spectra_raw, spectrum_i["spectrum"].reshape(1,-1)), axis = 0)

spectra_raw.shape
spectrum_pure_sc = spectra_raw.copy()
for i in range(spectra_raw.shape[0]):
        spectrum_pure_sc[i, :] = (spectra_raw[i, :] - np.min(spectra_raw[i, :])) / (np.max(spectra_raw[i, :]) - np.min(spectra_raw[i, :]))



#
colors =  [mcolors.CSS4_COLORS[key_i] for key_i in ['orangered','lime', 'saddlebrown', 'darkgoldenrod','olive','darkgreen','darkslategrey','dodgerblue','navy', 'darkviolet','darkmagenta','deeppink','crimson']]
# mixtures
DBcoms = range(1,14)# 13 components from 1 to 13

num = 1600
ts_size = 300
ratio_mae_all = []
#
num_pred_list_full = []

for num_keep in range(1,14):
    print(num_keep)
    spectrum_augB_00, ratios_aug0 = component_mixing(spectrum_pure_sc, num, num_keep)
    spectrum_augB_0, ratios_aug = spectrum_augB_00[-ts_size:,:],  ratios_aug0[-ts_size:,:]
    #
    num_keep_n_0 = num2words(num_keep)
    num_keep_n = num_keep_n_0[0].upper() + num_keep_n_0[1:].lower()
    #
    spectrum_mix_sc = spectrum_augB_0.copy() 
    for i in range(spectrum_augB_0.shape[0]):
            spectrum_mix_sc[i, :] = (spectrum_augB_0[i, :]-np.min(spectrum_augB_0[i, :])) / (np.max(spectrum_augB_0[i, :] )-np.min(spectrum_augB_0[i, :]))
    
    X = np.zeros((spectrum_mix_sc.shape[0]*spectrum_pure_sc.shape[0],2,1976))
    for p in range(spectrum_mix_sc.shape[0]):
        for q in range(spectrum_pure_sc.shape[0]):
            X[int(p*spectrum_pure_sc.shape[0]+q),0,:] = spectrum_mix_sc[p,:]
            X[int(p*spectrum_pure_sc.shape[0]+q),1,:] = spectrum_pure_sc[q,:]
    #
    spectrum_pure_scS = WhittakerSmooth_MAT(spectrum_pure_sc, lamb=1)
    spectrum_pure_scSB = airPLS_MAT(spectrum_pure_scS, lamb=10, itermax=10)
    spectrum_mix_scBS = WhittakerSmooth_MAT(spectrum_mix_sc, lamb=1)
    spectrum_mix_scBSB = airPLS_MAT(spectrum_mix_scBS, lamb=10, itermax=10)
    ##
    predic_ratios = np.zeros((spectrum_mix_scBSB.shape[0], 13))
    predic_comp = np.zeros((spectrum_mix_scBSB.shape[0], 13))
    gt_comp = np.zeros((spectrum_mix_scBSB.shape[0], 13))
    for cc in range(spectrum_mix_scBSB.shape[0]):
        com_gt_0 = ratios_aug.copy()[cc,:]
        thre_num = 0.000
        com_gt_0[com_gt_0 <= thre_num] = 0
        com_gt_0[com_gt_0 > thre_num] = 1
        #
        com = np.where(com_gt_0 ==1)[0].tolist()
        gt_comp[cc,:] = com_gt_0
        #
        X = spectrum_pure_scSB[com]
        coms = [DBcoms[com[h]] for h in range(len(com))]
    
        _, coefs_lasso, _ = enet_path(X.T, spectrum_mix_scBSB[cc,:], l1_ratio=0.96,
                                  positive=True)
        
        ratio = coefs_lasso[:, -1]
        ratio_sc = ratio.copy()
        for ss2 in range(ratio.shape[0]):
            ratio_sc[ss2]=ratio[ss2]/np.sum(ratio)
    
        print("cc = {}".format(cc))
        print('The',cc, 'spectra may contain {} components including:'.format(len(coms)),coms)
        print('The corresponding ratio is:', ratio_sc)
        ##
        predic_ratios[cc,com] = ratio_sc
    ##
    ratio_mae_i = np.mean(np.abs(ratios_aug - predic_ratios))
    print("{}, ratio_mae_i: {}".format(num_keep, ratio_mae_i))
    ratio_mae_all.append(ratio_mae_i)

