import pandas as pd
import time
import sys
import argparse

import psutil

from rdkit import Chem

from descriptors import wotan

from utils.files import *
from utils.descriptors import clean_decriptors, check_molec_struc

''' Help information about arguments if the script '''

parser=argparse.ArgumentParser(
    description='''Arguments for wotan.py script ''',
    epilog="""Output file with the same extension as input file will be: INPUTFILE_calculated """)
parser.add_argument('INPUTFILE', nargs='*', default=[1], help='It must be a valid file containing your SMILES of interest with "SMILES" text in the header.')
args=parser.parse_args()

ftypes = [
    ('Excel files',  '*.xlsx'),
    ('CSV files',    '*.csv'),
    ('TXT files',    '*.txt'),
    ('CAS files',    '*.cas'),
    ('SMILES files', '*.smi'),
    ('SDF files',    '*.sdf'),
]

desc_groups = [
    'AtomCentred',
    'Autocorrelation',
    'BurdenEigenvalues',
    'ConnectivityIndices',
    'Constitutional',
    'EdgeAdjacency',
    'Eigenvalues',
    'GETAWAY',
    'InformationIndices',
    'Rdkit3D',
    'Topological',
    'TopologicalCharge',
    'TwoDimensional',
    'WalkPathCounts',
    'FunctionalGroup',
]

class Wotan():
    def check_input(filepath):
        '''Auxiliary method for checking user input.
        '''

        data_df = load_dataframe(filepath, ftypes)
        return data_df

    def get_process_memory():
        process = psutil.Process(os.getpid())
        mi = process.memory_info()
        return mi.rss, mi.vms

    def format_bytes(bytes):
        if abs(bytes) < 1000:
            return str(bytes)+"B"
        elif abs(bytes) < 1e6:
            return str(round(bytes/1e3,2)) + "kB"
        elif abs(bytes) < 1e9:
            return str(round(bytes / 1e6, 2)) + "MB"
        else:
            return str(round(bytes / 1e9, 2)) + "GB"



    def calculate_wotan(filepath):
        '''Auxiliary method for calling the descriptor calculator (Wotan).
        '''
        print('[+] Reading input df')
        data_df = Wotan.check_input(filepath)
        print(data_df.head())
        print(data_df.shape)


        chatin = []

        print('[+] Preprocessing input df')
        molec_struc = [Chem.MolFromSmiles(smi) for smi in data_df['SMILES']]

        # to add y in the final df, but ensuring that only is added for correct SMILES
        y = list(data_df['y'])

        todrop = []

        for i, element in enumerate(molec_struc):
            if not element:
                todrop.append(i)

        short_y = [j for i, j in enumerate(y) if i not in todrop]

        # 2. Some CAS entries do not contain SMILES in the CACTUS server
        molec_struc, molec_struc_none = check_molec_struc(molec_struc)

        smiles = [Chem.MolToSmiles(mol) for mol in molec_struc]

        results = pd.DataFrame({'SMILES': smiles, 'y':short_y})

        results.to_csv(out_path, sep = ';', index=False, encoding='utf-8')

        dict_descriptos = {}

        for group in desc_groups:

            print('Calculating {}'.format(group))
            group = group + '_all'
            desc_fn = getattr(wotan, group)

            group_df = pd.DataFrame()
            start_time = time.time()
            for mol in molec_struc:
                group_df = pd.concat([group_df, desc_fn(mol)], axis=0, ignore_index=True)

            print('{} calculated in --- {} seconds ---'.format(group, time.time() - start_time))
            chatin.append('{} calculated in --- {} seconds ---'.format(group, time.time() - start_time))
            print('Saving {}'.format(group))
            chatin.append('Saving {}'.format(group))
            start_time = time.time()
            results = pd.read_csv(out_path, sep = ';', header=0, encoding='utf-8')
            results = pd.concat([results, group_df], axis=1)
            results.to_csv(out_path, sep = ';', index=False, encoding='utf-8')
            print('{} saved in --- {} seconds ---'.format(group, time.time() - start_time))
            chatin.append('{} saved in --- {} seconds ---'.format(group, time.time() - start_time))
        print('Descriptors succesfully calculated')
        print('Descriptors calculated in --- {} seconds ---'.format((time.time() - start_total_time) ))
        chatin.append('Descriptors calculated in --- {} seconds ---'.format((time.time() - start_total_time) ))


        print("Eliminated indexes", todrop)


if __name__ == '__main__':

    ####################### Calculate profile of execution #####################
    start_total_time = time.time()
    rss_before, vms_before = Wotan.get_process_memory()
    ############################################################################

    try:
        filepath = sys.argv[1]

    except IndexError:
        print('Invalid arguments. Please read help documentation.')
        exit()


    out_path = filepath.replace(os.path.basename(filepath).split(".")[0],
                                    os.path.basename(filepath).split(".")[0]
                                    + '-calculated_with_y')

    results = Wotan.calculate_wotan(filepath)

    print('And now our computation has ended')
    print('File created: \n ', out_path)

    ####################### Calculate profile of execution #####################
    elapsed_time = time.time() - start_total_time
    rss_after, vms_after = Wotan.get_process_memory()

    print("\n Profile: RSS: {:>8} | VMS: {:>8} | time: {:>8}"
        .format(Wotan.format_bytes(rss_after - rss_before),
                Wotan.format_bytes(vms_after - vms_before),
                elapsed_time))
    ############################################################################
