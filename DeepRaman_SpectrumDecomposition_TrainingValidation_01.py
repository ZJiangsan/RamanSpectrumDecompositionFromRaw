#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:49:23 2024

@author: nibio
"""



import glob
import os
import numpy as np
import random


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
###     data preparation for deepRaman model
spectra_raw_n_lb = spectra_raw_n[:] # the 13-component libary used in this study
X_data_TR = np.zeros((spectrum_augB_or_TR.shape[0]*spectra_raw_n_lb.shape[0], 2, 1976))
Y_data_TR = np.zeros(spectrum_augB_or_TR.shape[0]*spectra_raw_n_lb.shape[0])
for l_i in range(spectrum_augB_or_TR.shape[0]):
    # 
    label_l_i = ratios_aug_TR[l_i,:] 
    label_l_i[np.where(label_l_i>0)] = 1
    # 
    Y_data_TR[l_i*spectra_raw_n_lb.shape[0]:(l_i+1)*spectra_raw_n_lb.shape[0]] = label_l_i
    #
    for c_i in range(spectra_raw_n_lb.shape[0]): 
        X_data_TR[int(l_i*spectra_raw_n_lb.shape[0]+c_i),0,:] = spectrum_augB_or_TR[l_i,:]
        X_data_TR[int(l_i*spectra_raw_n_lb.shape[0]+c_i),1,:] = spectra_raw_n_lb[c_i,:]
###
spectra_raw_n_lb = spectra_raw_n[:]
X_data_TS = np.zeros((spectrum_augB_or_TS.shape[0]*spectra_raw_n_lb.shape[0], 2, 1976))
Y_data_TS = np.zeros(spectrum_augB_or_TS.shape[0]*spectra_raw_n_lb.shape[0])
for l_i in range(spectrum_augB_or_TS.shape[0]):
    # 
    label_l_i = ratios_aug_TS[l_i,:]
    label_l_i[np.where(label_l_i>0)] = 1
    #
    Y_data_TS[l_i*spectra_raw_n_lb.shape[0]:(l_i+1)*spectra_raw_n_lb.shape[0]] = label_l_i
    #
    for c_i in range(spectra_raw_n_lb.shape[0]):
        X_data_TS[int(l_i*spectra_raw_n_lb.shape[0]+c_i),0,:] = spectrum_augB_or_TS[l_i,:]
        X_data_TS[int(l_i*spectra_raw_n_lb.shape[0]+c_i),1,:] = spectra_raw_n_lb[c_i,:]

#
print(X_data_TR.shape)
print(X_data_TS.shape)
##
print(np.unique(Y_data_TR, return_counts = True))
print(np.unique(Y_data_TS, return_counts = True))



import torch.utils.data as udata
from torch.utils.data import DataLoader


class RamanDataset(udata.Dataset):
    def __init__(self, mode='train'):
        if (mode != 'train') & (mode != 'val'):
            raise Exception("Invalid mode!", mode)
        self.mode = mode
        if self.mode == 'train':
            spectra_raw_in = X_data_TR
        else:
            spectra_raw_in = X_data_TS
        
        self.length = spectra_raw_in.shape[0]
        
    def __len__(self):
        return self.length 
    def __getitem__(self, index):
        if self.mode == 'train':
            spectra_raw_in = X_data_TR
            spectra_ratio_in = Y_data_TR
        else:
            spectra_raw_in = X_data_TS
            spectra_ratio_in = Y_data_TS
        #
        data_spec_raw = np.array(spectra_raw_in[index-1,:,:])
        data_spec_ratio = np.array(spectra_ratio_in[index-1])
        #
        return data_spec_raw, data_spec_ratio

#
trainDataset = RamanDataset('train') 
testDataset = RamanDataset('test')
#
loader_train = DataLoader(dataset=trainDataset,
                                num_workers=0,  
                                batch_size=int(len(X_data_TR)/800),
                                shuffle=True,
                                pin_memory=True)

loader_test = DataLoader(dataset=testDataset,
                        num_workers=0, 
                        batch_size=int(len(X_data_TS)/800),
                        shuffle=True,
                        pin_memory=True)

# Model               
dataloaders = {
  'train': loader_train,
  'val': loader_test
}


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

        self.fc1 = nn.Linear(32 * (1 + 4 + 9 + 16), 1024)  # Adjust the size if necessary
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
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)




# Define model, loss function, and optimizer
deepRaman = DeepRamanModel().cuda()
criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(deepRaman.parameters(), lr=0.01)  # Reduced learning rate
#
###
## load the models if you have trained ones 
# resume_deepRaman = os.path.join(os.getcwd(), 'DeepRaman_RamanSpectrumDecomposition_01.pth') # it is the wrong name only
# checkpoint_deepRaman = torch.load(resume_deepRaman)
# deepRaman.load_state_dict(checkpoint_deepRaman['state_dict'])


epoch_init = 0
epoch_total = 350000001
#
accu_l = 0
##
LR = 0.01
loss_test_save = np.inf
loss_train_save = np.inf
accuray_test_save = 0
accuray_train_save = 0
epoch_save = 0
accu_l = 0
##
loss_test_full = []
loss_train_full = []
accy_test_full = []
accy_train_full = []
for epoch in range(epoch_init, epoch_total):
    # evaluation
    deepRaman.eval()
    with torch.no_grad():
        loss_test = 0
        test_b = 0
        for inputs, labels in dataloaders["val"]:
            input_test = inputs.squeeze().cuda().float()
            #
            label_test = labels.squeeze().cuda().float()
            
            outputs_test = deepRaman(input_test)
            
            # 
            loss_test_0 = criterion(outputs_test, label_test)
            loss_test +=loss_test_0.item()
            #
            if test_b ==0:
                pred_test_out = outputs_test.clone()
                gt_test_out = label_test.clone()
            else:
                pred_test_out = torch.cat((pred_test_out, outputs_test.clone()), 0)
                gt_test_out = torch.cat((gt_test_out, label_test.clone()), 0)
            
            test_b+=1
        #
        loss_test_out = loss_test/len(dataloaders["val"])
        loss_test_full.append(loss_test_out)
        # print('current test: loss of {} at epoch of {}'.format(loss_test_out, epoch))
        accuray_test = calculate_accuracy(pred_test_out, gt_test_out).item()
        accy_test_full.append(accuray_test)
        # print('current test: accuracy of {} at epoch of {}'.format(accuray_test, epoch))
    # training
    deepRaman.train()
    #
    optimizer = torch.optim.Adam(deepRaman.parameters(), lr=LR, weight_decay=1e-4)  # Reduced learning rate
    #
    loss_train = 0
    train_b = 0
    for inputs, labels in dataloaders["train"]:
        input_train = inputs.squeeze().cuda().float()
        #
        label_train = labels.squeeze().cuda().float()
        
        optimizer.zero_grad()

        outputs = deepRaman(input_train)
        # Compute the loss
        loss = criterion(outputs, label_train)
        loss_train +=loss.item()
        #
        # append for final accuracy calcualtion
        if train_b ==0:
            pred_train_out = outputs.clone()
            gt_train_out = label_train.clone()
        else:
            pred_train_out = torch.cat((pred_train_out, outputs.clone()), 0)
            gt_train_out = torch.cat((gt_train_out, label_train.clone()), 0)
        # 
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        #
        train_b +=1
    #
    loss_train_out = loss_train/len(dataloaders["train"])
    loss_train_full.append(loss_train_out)
    # print('current train: loss of {} at epoch of {}'.format(loss_train_out, epoch))
    accuray_train = calculate_accuracy(pred_train_out, gt_train_out).item()
    accy_train_full.append(accuray_train)
    # print('current train: accuracy of {} at epoch of {}'.format(accuray_train, epoch))
    ##
    accu_l +=1
    if loss_test_save>loss_test_out:
        accu_l = 0
        lr_save_loss = LR
        epoch_save = epoch
        loss_test_save = loss_test_out
        accuray_test_save = accuray_test
        loss_train_save = loss_train_out
        accuray_train_save = accuray_train
        save_checkpoint_sp(os.getcwd(), 10000, lr_save_loss, "DeepRaman", deepRaman, optimizer) # 32233 is good enough
    ##
    # 
    print("")
    print("Current Testing-- loss {} and accuracy {},at epoch: {} and lr:{}".format(round(loss_test_out,4),round(accuray_test,4),epoch, LR))
    print("Current Training-- loss: {} and accuracy {} at epoch: {} and lr:{}".format(round(loss_train_out,4),round(accuray_train,4), epoch, LR))
    print("")
    print("the saved values")
    print("the smallest loss on test = {} at epoch of {} and LR of {}".format(loss_test_save,epoch_save,lr_save_loss))
    print("the corresponding accuracy on test = {}".format(accuray_test_save))
    print("the smallest loss on train = {}".format(loss_train_save))
    print("the corresponding accuracy on train = {}".format(accuray_train_save))
    print("current accu_l = {} at epoch of {}".format(accu_l, epoch))
    if accu_l > 10:
        accu_l = 0
        LR = LR* 0.98



