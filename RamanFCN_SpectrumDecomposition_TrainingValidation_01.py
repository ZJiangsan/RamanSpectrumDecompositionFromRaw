#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:28:59 2024

@author: nibio
"""



import glob
import os
import numpy as np
import chardet
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import torch.utils.data as udata
from torch.utils.data import DataLoader



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


##
def l_2_1_loss(im_true, im_fake):
    n, m = im_true.shape
    error = ((((im_fake-im_true).pow(2)).sum(1)).pow(1/2)).sum(0)
    return error/n/m
#
def save_checkpoint_sp(model_path, epoch, LR, name, model, optimizer_gen):
    """Save the checkpoint."""
    state = {
            'epoch': epoch,
             'lr': LR,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_gen.state_dict(),
            }
    torch.save(state, os.path.join(model_path, '{}_RamanSpectrumDecomposition_01.pth'.format(name)))
#
def calculate_accuracy(predictions, labels):
    preds = torch.round(predictions)  # Convert probabilities to binary predictions (0 or 1)
    correct = (preds == labels).float()  # Convert to float for division
    accuracy = correct.sum() / len(correct)
    return accuracy
#

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


## separate the training and validation datasets

num = 1600 # how many are you going to augment
tr_size = 1000
for num_keep_i in range(2,14): # from 2 to 13 componnets combinations
    print("num_keep_i = {}".format(num_keep_i))
    spectrum_augB_in_0_i, ratios_aug_in_i = component_mixing(spectra_raw_n, num, num_keep_i)

    if num_keep_i ==2:
        spectrum_augB_0_TR = spectrum_augB_in_0_i[:tr_size,:]
        ratios_aug_TR = ratios_aug_in_i[:tr_size,:]
        ###
        spectrum_augB_0_TS = spectrum_augB_in_0_i[tr_size:tr_size+300,:]
        ratios_aug_TS = ratios_aug_in_i[tr_size:tr_size+300,:]
    else:
        spectrum_augB_0_TR = np.concatenate((spectrum_augB_0_TR, spectrum_augB_in_0_i[:tr_size,:] ), axis = 0)
        ratios_aug_TR = np.concatenate((ratios_aug_TR, ratios_aug_in_i[:tr_size,:]), axis = 0)
        ##
        spectrum_augB_0_TS = np.concatenate((spectrum_augB_0_TS, spectrum_augB_in_0_i[tr_size:tr_size+300,:] ), axis = 0)
        ratios_aug_TS = np.concatenate((ratios_aug_TS, ratios_aug_in_i[tr_size:tr_size+300,:]), axis = 0)


## normalize the newly created spectra and baseline
spectrum_augB_or_TR = spectrum_augB_0_TR.copy()
for i in range(spectrum_augB_0_TR.shape[0]):
        spectrum_augB_or_TR[i, :] = (spectrum_augB_0_TR[i, :]-np.min(spectrum_augB_0_TR[i, :])) / (np.max(spectrum_augB_0_TR[i, :])-np.min(spectrum_augB_0_TR[i, :]))
###
spectrum_augB_or_TS = spectrum_augB_0_TS.copy()
for i in range(spectrum_augB_0_TS.shape[0]):
        spectrum_augB_or_TS[i, :] = (spectrum_augB_0_TS[i, :]-np.min(spectrum_augB_0_TS[i, :])) / (np.max(spectrum_augB_0_TS[i, :])-np.min(spectrum_augB_0_TS[i, :]))


###########################
#
print(spectrum_augB_or_TR.shape)
print(ratios_aug_TR.shape)

print(spectrum_augB_or_TS.shape)
print(ratios_aug_TS.shape)


# Generate some example data

class RamanDataset(udata.Dataset):
    def __init__(self, mode='train'):
        if (mode != 'train') & (mode != 'test'):
            raise Exception("Invalid mode!", mode)
        self.mode = mode
        if self.mode == 'train':
            spectra_raw_in = spectrum_augB_or_TR

        else:
            spectra_raw_in = spectrum_augB_or_TS
        
        self.length = spectra_raw_in.shape[0]
        
    def __len__(self):
        return self.length 
    def __getitem__(self, index):
        if self.mode == 'train':
            spectra_raw_in = spectrum_augB_or_TR
            spectra_ratio_in = ratios_aug_TR

        else:
            spectra_raw_in = spectrum_augB_or_TS
            spectra_ratio_in = ratios_aug_TS
        #
        data_spec_raw = np.array(spectra_raw_in[index-1,:]).reshape(1,-1)
        data_spec_ratio = np.array(spectra_ratio_in[index-1,:]).reshape(1,-1)
        #
        return data_spec_raw, data_spec_ratio

## data preparation   (only need to run at initiation)

trainDataset = RamanDataset('train')  ## here not the training data but the whole data set for this work
testDataset = RamanDataset('test')  ## here not the training data but the whole data set for this work

# len(trainDataset)

# Data Loader (Input Pipeline)
loader_train = DataLoader(dataset=trainDataset,
                                # sampler=train_sampler,
                                num_workers=0,  
                                batch_size=len(spectrum_augB_or_TR), #int(num*0.8), #len(spectra_clean_TR),
                                shuffle=False,
                                pin_memory=True)

loader_test = DataLoader(dataset=testDataset,
                        # sampler=validation_sampler,
                        num_workers=0, 
                        batch_size=len(spectrum_augB_or_TS), #len(spectra_clean_TS),
                        shuffle=False,
                        pin_memory=True)

spectrum_augB_or_TS.shape
# Model               
dataloaders = {
  'train': loader_train,
  'val': loader_test
}

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
###




##.                   initiate all the networks
end_epoch = 300000
LR = 1e-4
hsize = 13
#
ramanFCN = RamanFCN().cuda().float()
optimizer_ramanFCN = torch.optim.Adam(ramanFCN.parameters(), lr=LR, weight_decay=1e-4)
##


## load the models if you have trained ones 
# resume_ramanFCN = os.path.join(os.getcwd(), 'ramanFCN_RamanSpectrumDecomposition_01.pth')
# checkpoint_ramanFCN = torch.load(resume_ramanFCN)
# ramanFCN.load_state_dict(checkpoint_ramanFCN['state_dict'])




#
test_save_loss = np.inf
train_save_0_loss = np.inf
#
loss_test_full = []
loss_train_full = []

component_library = torch.tensor(spectra_raw_n).float().cuda()

# Model
n_st = 2
epoch_init = 0
epoch_total = 350000001
#
accu_l = 0
##
for epoch in range(epoch_init, epoch_total):
    #
    ramanFCN.eval()   # Set model to evaluate mode
    # decoder.eval()
    with torch.no_grad():
        loss_out_test_0 = 0
        #
        test_b = 0
        for inputs, labels in dataloaders["val"]:
            input_test = inputs.squeeze().cuda().float() 
            label_test = labels.squeeze().cuda().float()
            #
            out_test_v = ramanFCN(input_test)
            out_test_v = out_test_v.clamp(1e-8, 1e+8)
            
            out_test_s = out_test_v/out_test_v.sum(1, keepdim = True)
            out_test_s = out_test_s.clamp(1e-8, 1.0-1e-8)
            ##
            loss_out_test_0_i = l_2_1_loss(label_test+ torch.tensor(1e-8).cuda(), out_test_s+ torch.tensor(1e-8).cuda())
            ##
            loss_out_test_0 += loss_out_test_0_i.item()
            # convert to mixture spectra for visualization
            out_test_de_0 = torch.matmul(out_test_s, component_library)
            out_test_de_0 = out_test_de_0.clamp(1e-8, 1e+8)
            #
            out_test_de = (out_test_de_0 - out_test_de_0.min(1, keepdim = True).values)/(out_test_de_0.max(1,keepdim =True).values - out_test_de_0.min(1, keepdim = True).values)
            #######
            if (epoch) % 400 == 0:
                spec_or_npB = input_test[:,:].detach().cpu().squeeze().numpy()
                spec_de_np = out_test_de.detach().cpu().squeeze().numpy() 
                #
                num_n = int(num/5)
                x0 =np.linspace(1, 1976, 1976, endpoint=True) 
                for p_i in range(2,13):
                    #
                    plt.figure(figsize=(12,8))
                    y00 = spec_or_npB[(p_i-n_st)*num_n,:]
                    y02 = spec_de_np[(p_i-n_st)*num_n,:]
                    plt.plot(x0, y00, color = 'r', linestyle = ':', label = "Spec_gt")
                    plt.plot(x0, y02, color = 'g', linestyle = '-', label = "Spec_rec")
                    plt.legend()
                    plt.show()
                #
                ##.   Training error maps
                ratio_or_np = label_test[:,:].detach().cpu().squeeze().numpy() 
                ratio_de_np = out_test_s.detach().cpu().squeeze().numpy()
                # 
                ratio_ls = [(r_s-n_st)*num_n for r_s in range(n_st,13)]
                #
                plt.figure()
                plt.imshow(ratio_or_np[ratio_ls,:],vmin=0,vmax=1); plt.title("gt,epoch={}".format(epoch)) 
                plt.show()
                #
                plt.figure()
                plt.imshow(ratio_de_np[ratio_ls,:],vmin=0,vmax=1); plt.title("Rec,epoch={}".format(epoch))
                plt.show()
                #
                ratio_ae = np.abs(ratio_or_np - ratio_de_np)
                # 
                plt.figure()
                plt.imshow(ratio_ae[ratio_ls,:],vmin=0,vmax=1); plt.title("AE train, epoch={}".format(epoch))
                plt.show()
                    ##
            test_b +=1
        #
        loss_out_test = loss_out_test_0/len(dataloaders["val"])
        loss_test_full.append(loss_out_test)
    ###########################
    ramanFCN.train() 
    optimizer_ramanFCN = torch.optim.Adam(ramanFCN.parameters(), lr=LR, weight_decay=1e-4)
    ##
    loss_out_train_0 = 0
    #
    for inputs, labels in dataloaders["train"]:
        input_train = inputs.squeeze().cuda().float() 
        #
        label_train = labels.squeeze().cuda().float()
        #
        optimizer_ramanFCN.zero_grad()
        #
        out_train_v = ramanFCN(input_train)
        out_train_v = out_train_v.clamp(1e-8, 1e+8)
        #
        out_train_s = out_train_v/out_train_v.sum(1, keepdim = True)
        out_train_s = out_train_s.clamp(1e-8, 1.0-1e-8)
        #
        ##
        loss_out_train_0_i = l_2_1_loss(label_train + torch.tensor(1e-8).cuda(), out_train_s + torch.tensor(1e-8).cuda())
        ##
        loss_out_train_0 += loss_out_train_0_i.item()
        ##
        loss_out_train_final = 1*loss_out_train_0_i
        ###
        loss_out_train_final.backward()
        #
        optimizer_ramanFCN.step()
    #
    loss_out_train = loss_out_train_0/len(dataloaders["train"])
    loss_train_full.append(loss_out_train)
    ##
    accu_l +=1
    if test_save_loss > loss_out_test:
        accu_l = 0
        test_save_loss = loss_out_test
        #
        train_save_loss = loss_out_train
        lr_save_loss = LR
        epoch_save_loss = epoch
        #
        save_checkpoint_sp(os.getcwd(), 10000, lr_save_loss, "ramanFCN", ramanFCN, optimizer_ramanFCN) # 32233 is good enough

    print("")
    print("Current Testing-- spectrum: {},at epoch: {} and lr:{}".format(round(loss_out_test,6), epoch, LR))
    print("Current Training-- spectrum: {} at epoch: {} and lr:{}".format(round(loss_out_train,6), epoch, LR))
    print("")
    print("Saved Testing loss {}".format(test_save_loss))
    print("Testing--last saved spectrum: {} at epoch: {} and lr:{}".format(round(test_save_loss,6), epoch_save_loss, lr_save_loss))
    print("Training--last saved spectrum: {} at epoch: {} and lr:{}".format(round(train_save_loss,6), epoch_save_loss, lr_save_loss))
    print("")
     
    if accu_l > 10:
        accu_l = 0
        LR = LR* 0.98







