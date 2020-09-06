import os.path
from os import listdir
from os.path import isfile, join
from web_scraping import cas_to_smiles
import csv
import pandas as pd

path = '../data/Models_not_in_REACH/phototoxicity/'

dataset = pd.read_csv(path + 'photoxicity_dataset.txt', sep = '\t', header = 0)

cas_list = dataset['CAS'].tolist()

smiles_list = cas_to_smiles(cas_list)

with open('photoxicity_dataset_smiles.txt', 'w') as f:
    for item in smiles_list:
        f.write("%s\n" % item)
