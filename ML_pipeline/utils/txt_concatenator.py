import os.path
from os import listdir
from os.path import isfile, join
from web_scraping import cas_to_smiles
import csv

path = '../data/Toxicology/rat_LC50_inhalation/'

filenames = [f for f in listdir(path) if isfile(join(path, f))]

cas_list = [os.path.splitext(cas)[0] for cas in filenames]

with open(path + 'rat_LC50_inhalation_dataset.txt', 'w') as outfile:
    for fname in filenames:
        with open(path + fname) as f:
            for line in f:
                outfile.write(f.read())
