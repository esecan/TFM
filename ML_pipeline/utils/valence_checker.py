import numpy as np
import pandas as pd

import time

from rdkit import Chem

# Script to find the smiles with valence errors in order to delete them

def valence_checker(in_path, out_path):

    test_data = pd.read_csv(in_path, header=0, sep='\t')

    smiles = test_data['SMILES']

    mol_list = []
    mol_ok_idx = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        try:
            Chem.SanitizeMol(mol)
            mol_list.append(mol)
            mol_ok_idx.append(list(smiles).index(smi))
        except Exception as error:
            # mol_error_idx.append(smiles.index(smi))
            print('[ERROR] {}: {}'.format(error, smi))

    test_data = test_data.iloc[mol_ok_idx, :]

    test_data.to_csv(out_path, sep='\t', index=False, encoding='utf-8')




if __name__ == '__main__':

    start_time = time.time()

    data_dir = 'datasets/'
    in_path = data_dir + 'log_kow-descriptors-test.txt'
    out_path = data_dir + 'log_kow-descriptors-test_valence_curated.txt'

    valence_checker(in_path, out_path)

    print('--- %s minutes ---' % round((time.time() - start_time) / 60, 3))
