import pandas as pd
import time
import sys
import argparse

import psutil
import multiprocessing
from multiprocessing import Process

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



class Calculate(Process):
    def __init__ (self, input_df,i,ini,fin,desc_groups,return_dict):

        Process.__init__ (self)
        self.i = i
        self.input_df = input_df
        self.ini = ini
        self.fin = fin
        self.desc_groups = desc_groups
        self.return_dict = return_dict

    def run (self):

        self.df = self.input_df.iloc[self.ini:self.fin,:]
        molec_struc = [Chem.MolFromSmiles(smi) for smi in self.df['SMILES']]
        smiles = [Chem.MolToSmiles(mol) for mol in molec_struc]
        y = list(self.df['y'])
        self.results = pd.DataFrame({'SMILES': smiles, 'y':y})

        for group in desc_groups:
            start_time = time.time()
            group = group + '_all'
            desc_fn = getattr(wotan, group)
            group_df = pd.DataFrame()

            for mol in molec_struc:
                group_df = pd.concat([group_df, desc_fn(mol)], axis=0, ignore_index=True)

            print('Custer {}: {} calculated in --- {} seconds ---'.format(self.i, group, time.time() - start_time))
            self.results = pd.concat([self.results, group_df], axis=1)

        self.return_dict[self.i] = self.results


class Wotan():

    def check_input(filepath):
        '''Auxiliary method for checking user input.
        '''
        data_df = load_dataframe(filepath, ftypes)
        return data_df

    def preprocess_input_df(data_df):

        smiles = list(data_df['SMILES'])
        molec_struc = [Chem.MolFromSmiles(smi) for smi in data_df['SMILES']]
        y = list(data_df['y'])

        todrop = []
        mols_dropped = []

        for i, element in enumerate(molec_struc):
            if not element:
                todrop.append(i)

        if len(todrop) > 0:
            for j in todrop:
                mols_dropped.append(smiles[j])

        del smiles

        short_y = [j for i, j in enumerate(y) if i not in todrop]

        # 2. Some CAS entries do not contain SMILES in the CACTUS server
        molec_struc, molec_struc_none = check_molec_struc(molec_struc)

        smiles = [Chem.MolToSmiles(mol) for mol in molec_struc]

        input_df = pd.DataFrame({'SMILES': smiles, 'y':short_y})

        return input_df, mols_dropped

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
                                    + '-paralel_calculated_with_y')

    print('[+] Reading input df')
    data_df = Wotan.check_input(filepath)
    print(data_df.head())
    print(data_df.shape)

    print('[+] Preprocessing input df')
    # It eliminates incorrect molecules
    input_df, mols_dropped = Wotan.preprocess_input_df(data_df)

    input_df.to_csv(out_path, sep = ';', index=False, encoding='utf-8')

    shape = input_df.shape

    print('[+] Splitting input df for parallel calculation')

    cores = multiprocessing.cpu_count()

    NUM_PROCESOS = int(cores/2)

    interv = int(shape[0]/NUM_PROCESOS)

    jobs = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(NUM_PROCESOS):

        if i == NUM_PROCESOS-1:
            ini =  i*interv
            fin = shape[0]
        else:
            ini =  i*interv
            fin = (i+1)*interv

        jobs.append(Calculate(input_df,i,ini,fin,desc_groups,return_dict))
        jobs[i].start()

    for job in jobs:
        job.join()


    final_df = pd.DataFrame()

    for i in range(NUM_PROCESOS):
        final_df = pd.concat([final_df,return_dict[i]], axis=0, sort=False)

    final_df.to_csv(out_path, sep = ';', index = False)


    if len(mols_dropped) > 0:
        print('\nThe following molecules are incorrect and have been eliminated:')
        for smile in mols_dropped:
            print(smile)


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
