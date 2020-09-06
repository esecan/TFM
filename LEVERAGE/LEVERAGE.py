# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:14:24 2020

@author: EvaSC
"""


import os
import math

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from rdkit import Chem

def file_checkpoint():
    flag_fc = True

    allowed_fc = ['y','Y','n','N']

    file_checkpoint = input("Continue (Y/n)?") or "y"
    while flag_fc:
        if file_checkpoint in allowed_fc:
            flag_fc = False
        else:
            file_checkpoint = input('\nIncorrect input. Continue (Y/n)?' or "y")
            continue

    if file_checkpoint == 'n' or file_checkpoint == 'N':
        print('\nThanks for using HYGIEIA!')
        exit()



print('\n#########################################################################'
        + '\n##################### WELCOME TO LEVERAGE script ########################'
        + '\n#########################################################################'
        + '\nThis script will: \n'
        + ' \t- plot the William\'s plot for yout data \n'
        + ' \t- inform you about outlier molecules\n')


PATH = input('Please input your PATH (enter to: "../data/"): ') or "../data/"
MODEL = input('Please input your MODEL NAME (enter to: avian_reproduction_toxicity): ') or "avian_reproduction_toxicity"
SEP = '\t'
INPUT_FILE = '{}-descriptors-train.txt'.format(MODEL)
INPUT_FILE_train = '{}-descriptors-test.txt'.format(MODEL)

print('The following files, separated by "\t", located in "{}" folder are needed'.format(PATH))
print('\t- "{}-descriptors-train.txt"'.format(MODEL))
print('\t- "{}-descriptors-test.txt"'.format(MODEL))
file_checkpoint()



train_data = pd.read_csv(
        PATH + INPUT_FILE,
        sep = SEP,
        header=0,
        encoding='latin'
        )

test_data = pd.read_csv(
        PATH + INPUT_FILE_train,
        sep = SEP,
        header=0,
        encoding='latin'
        )




###############################################################################
################################# LEVERAGES ###################################
###############################################################################

############################### train leverages ###############################

descriptors_train = train_data.iloc[:, 2:-1]
descriptors_train.head()

# scale 

scaler = StandardScaler()
scaler.fit(descriptors_train)
X = scaler.transform(descriptors_train)

# create descriptor matrix

tranpX = np.transpose(X)
xtx = np.dot(tranpX,X)
invxtx = np.linalg.inv(xtx)

# calculate h value for each row:

leverages = []

for row in X:
    transposed_row = np.transpose(row)
    leverage = np.dot(np.dot(row, invxtx), transposed_row)
    leverages.append(leverage)


# calculate leverage warning:
p = X.shape[1]
n = X.shape[0]
warning_leverage = 3*(p/n)

############################### test leverages ################################

descriptors_test = test_data.iloc[:, 2:-1]

# scale
Y = scaler.transform(descriptors_test)

# calculate h value for each row:
leverages_test = []

for row in Y:
    transposed_row = np.transpose(row)
    leverage_test = np.dot(np.dot(row, invxtx), transposed_row)
    leverages_test.append(leverage_test)


#%%

###############################################################################
############################### STUDENTIZED RESIDUALS #########################
###############################################################################

############################### train residuals ###############################

y_obs = train_data['y']
y_pred = train_data['predicted']

residue = y_obs - y_pred
residues = residue.tolist()

media = sum(residues)/len(y_obs)


var = sum((residue-media)**2)/n

sqrt_var = math.sqrt(var)

se_regs = []

for lev in leverages:
    se_reg = sqrt_var*((1-lev)**0.5)
    se_regs.append(se_reg)
    

studentized_residuals = residue/se_regs

st_res = studentized_residuals.to_list()

############################### test residuals ################################

y_obs_test = test_data['y']
y_pred_test = test_data['predicted']

   
residue_test = y_obs_test - y_pred_test

residues_test = residue_test.tolist()

media_test = sum(residues_test)/len(y_obs_test)

var_test = sum((residue_test-media_test)**2)/n

sqrt_var_test = math.sqrt(var_test)
    
se_regs_test = []

for lev_test in leverages_test:
    se_reg_test = sqrt_var_test*((1-lev_test)**0.5)
    se_regs_test.append(se_reg_test)
    


studentized_residuals_test = residue_test/se_regs_test

st_res_test = studentized_residuals_test.to_list()





#%%

###############################################################################
######################################## PLOT #################################
###############################################################################


fig, ax = plt.subplots()

ax.scatter(leverages, st_res, s=5, color="blue", label="h")
ax.scatter(leverages_test, st_res_test, s=5, color="red", label="h")

#lines
plt.axhline(y = -3, color="red") # studentized residuals lower threshold
plt.axhline(y = 3, color="red") # studentized residuals upper threshold
ax.axvline(x  =warning_leverage,color="blue") # leverage threshold


# plt.set(xlim=(0, 0.5), ylim=(-8, 8))
ax.set_xlim(0, max(leverages)+0.5)
ax.set_ylim(min(st_res)-2, max(st_res)+2)

ax.set_xlabel('Leverage', fontsize=10)
ax.set_ylabel('Standarized Residues', fontsize=10) 

plt.show()


#%%
###############################################################################
############################# OUTLIERS MOLECULES ##############################
###############################################################################

print('\nMolecules that are outside in the train set by leverage')
outliers_train = []

for i,lev in enumerate(leverages):
    if lev > warning_leverage:
        outliers_train.append(i)
        # print(i,lev)


for i,row in train_data.iterrows():
    if i in outliers_train:
        print(row[0])

print('\nMolecules that are outside in test set by leverage')
outliers_test = []

for i,lev in enumerate(leverages_test):
    if lev > warning_leverage:
        outliers_test.append(i)

for i,row in test_data.iterrows():
    if i in outliers_test:
        print(row[0])

print('\nMolecules that are outside in the train set by residues')
outliers_train_by_res = []


for i,st_res_ in enumerate(st_res):
    if st_res_ > 3 or st_res_ < -3:
        outliers_train_by_res.append(i)

for i,row in train_data.iterrows():
    if i in outliers_train_by_res:
        print(row[0])

print('\nMolecules that are outside in the test set by residues')
outliers_test_by_res = []

for i,st_res_test_ in enumerate(st_res_test):
    if st_res_test_ > 3 or st_res_test_ < -3:
        outliers_test_by_res.append(i)

for i,row in train_data.iterrows():
    if i in outliers_test_by_res:
        print(row[0])        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        