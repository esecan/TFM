# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 08:48:01 2020

@author: EvaSC
"""

import pandas as pd

import os

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


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
        + '\n####################### WELCOME TO HYGIEIA script #######################'
        + '\n#########################################################################'
        + '\nThis script will process your dataframe and: \n'
        + ' \t- eliminate incorrect molecules\n'
        + ' \t- clean salts\n'
        + ' \t- eliminate inorganic and organometallic compounds\n')


PATH = input('Please input your PATH (enter to: "../data/"): ') or "../data/"
MODEL = input('Please input your MODEL NAME (enter to: avian_reproduction_toxicity): ') or "avian_reproduction_toxicity"
SEP = ';'
INPUT_FILE = '{}-preprocessed.csv'.format(MODEL)

print('A file located in "{}" folder is needed'.format(PATH))
print('This file must be called: "{}-preprocessed.csv"'.format(MODEL))
file_checkpoint()



dataset_all = pd.read_csv(
        PATH + INPUT_FILE,
        sep = SEP,
        header=0,
        encoding='latin'
        )




dataset_smiles_y = dataset_all[['SMILES','y']]



##############################################################################
######################### STEP 1: smile sanitization #####################
##############################################################################

print('[+] Sanitizing smiles ')

san_smiles = []
incorrect_smiles = []

for smi in dataset_all['SMILES']:
    try:
        sanitized_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        san_smiles.append(sanitized_smi)
    except:
        san_smiles.append('-')
        incorrect_smiles.append(smi)

dataset_smiles_y.insert(1,'SAN_SMILES',san_smiles)


rows_to_drop_by_incorrect_smiles = set()

for i, row in dataset_smiles_y.iterrows():
    for j, value in  enumerate(row.iteritems()):
        if j == 0 and value[1] in incorrect_smiles:
            rows_to_drop_by_incorrect_smiles.add(i)

rows_to_drop_by_incorrect_smiles = list(rows_to_drop_by_incorrect_smiles)

df_clean_by_sanit = dataset_smiles_y.drop(dataset_smiles_y.index[rows_to_drop_by_incorrect_smiles])

if len(rows_to_drop_by_incorrect_smiles) >0:
    print('\tYour dataset had {} incorrect molecules. They have been eliminated.:\n'.format(len(rows_to_drop_by_incorrect_smiles)))
    for inco_smi in rows_to_drop_by_incorrect_smiles:
        print('\t\t',dataset_smiles_y.iloc[11]['SMILES'],'\n')
else:
    print('\tCongratulations, your dataset has not incorrect smiles.')
    

##############################################################################
############################ STEP 2: salt elimination ########################
##############################################################################

print('[+] Eliminating salts ')

withoutsalts = []

for smi in df_clean_by_sanit['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    remover = SaltRemover(defnData='[Na,Cl,K,O,OH,Fe,F,H,Al,Mg,Co,Ti,NH4,Mn,Si,Ca,Au,I,Hg,Mo,Zn,Br,Ag,Sr,Cu,Bi,S,Li,NH3,He,Y,Ar,Ba,La]')
    mol = remover.StripMol(mol)
    smiles_new = Chem.MolToSmiles(mol)
    smiles_new = smiles_new.replace('.[H+]', '').replace('[H+].', '') # because saltremover do not eliminate water
    withoutsalts.append(smiles_new)




df_clean_by_sanit.insert(2,'W/O SALTS',withoutsalts)


prompt = []

for smile_with, smile_without in zip(df_clean_by_sanit['SAN_SMILES'],df_clean_by_sanit['W/O SALTS']):
    if smile_with != smile_without:
        prompt.append('\t{} --> {}'.format(smile_with,smile_without))

if len(prompt) != 0:
    print('\tYour dataset had salts, that have been eliminated:\n')
    for messaje in prompt:
     print(messaje)
else:
    print('\tCongratulations, your dataset has not salts.')


#%%
##############################################################################
################ STEP 3: eliminate inorganic and organometallics #############
##############################################################################

print('[+] Eliminating inorganic and organometallic compounds')

not_allowed = ['Al', 'Fe', 'Ti', 'Pt', 'Pb', 'Si', 'As', 'Ni', 'Te', 'Hg', 'Cu','Se', 'se', 'Cd', 'Sn', 'Be', 'Mg', 'Mn', 'Cr', 'Tl', 'Zn', 'Au', 'Ag', 'Ge']


organometalics_andbad = set()
check_for_Boro = []
bromo = []


for a in df_clean_by_sanit['W/O SALTS']:

   
    for notal in not_allowed:
        if notal in a:
            organometalics_andbad.add(a)
        
    if not 'C' in a.upper():
        organometalics_andbad.add(a)
    
    if 'B' in a:

        check_for_Boro.append(a)

    if 'Br' in a:

        bromo.append(a)
        
 
 
for b in check_for_Boro:
    if b not in bromo:
        organometalics_andbad.add(b)


if len(organometalics_andbad) >0:
    print('\nYour dataset had {} incorrect molecules. They have been eliminated.'.format(len(organometalics_andbad)))
    for incorrect_smile in organometalics_andbad:
        print('\t',incorrect_smile)
else:
    print('\n\tCongratulations, your dataset has not inorganic and organometallic compounds.')
    
rows_to_drop = set()





for i, row in dataset_smiles_y.iterrows():
    for j, value in  enumerate(row.iteritems()):
        if value[1] in organometalics_andbad:
            rows_to_drop.add(i)

rows_to_drop = list(rows_to_drop)

df_clean = df_clean_by_sanit.drop(df_clean_by_sanit.index[rows_to_drop])

df_clean = df_clean[['W/O SALTS','y']]

df_clean.columns = ['SMILES','y']

print('\nA new file has been generated:\n')

print('{}.csv'.format(PATH+MODEL))
df_clean.to_csv('{}.csv'.format(PATH+MODEL),sep=';', index=False)



print('Your input dataset has {} molecules.'.format(dataset_all.shape[0]))

print('Your output dataset has {} molecules.'.format(df_clean.shape[0]))

