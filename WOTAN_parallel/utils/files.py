import os

import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools

# TODO: Replace with check_path from Drug-Discovery

def check_path(path_to_check, create=False, filepath=False):
    '''Checks if a directory exists and creates it otherwise.
    '''
    original_path = path_to_check

    # Create directory or directory containing the file
    if create:
        if filepath:
            path_to_check = os.path.split(path_to_check)[0]

        if not os.path.exists(path_to_check):
            try:
                os.makedirs(path_to_check)

            except Exception as err:
                print('[ERROR] Creating directories: {}'.format(err))

                return None

        return original_path

    # Check whether directory or files exists
    if os.path.exists(path_to_check):
        return original_path
    else:
        return None


def clean_string(string, chars=[' ', '\n']):
    '''Clean string from the desired characters.
    '''
    for ch in chars:
        string = string.replace(ch, '')

    return string


def check_ftype(input_data):
    '''Checks input data and discriminates between files or manual entries.
    '''
    ftype = input_data.split('.')

    # If there is more than one element after splitting, it is a path
    if len(ftype) > 1:
        return input_data, ftype[-1].lower()

    # Otherwise, set ftype as None and clean the user input
    else:
        return clean_string(input_data), None


def load_dataframe(input_data, accepted_ftypes):
    '''Loads a dataframe from .xlsx, .csv, .txt, .smi or .cas files.
    '''
    input_data, ftype = check_ftype(input_data)

    if ftype:
        # Check if ftype is in the accepted ftypes in app.py
        accepted_ftypes = [tupla[1].replace('*.', '')
                           for tupla in accepted_ftypes]
        if ftype not in accepted_ftypes:
            print('[ERROR] Format not supported')
            return -999

    if ftype == 'xlsx':
        return pd.read_excel(input_data, header=0)

    elif ftype == 'csv':
        return pd.read_csv(input_data, sep=';', header=0)

    elif ftype == 'txt':
        return pd.read_csv(input_data, sep='\t', header=0)

    elif ftype == 'sdf':
        return load_sdf(input_data)

    elif ftype == 'smi' or ftype == 'cas':
        return load_raw_list(input_data, ftype)

    elif not ftype:
        return load_user_input(input_data)


def load_sdf(path):
    '''Loads data from SDF files.
    '''
    # No direct structure (png?) but labels and names
    dataset = PandasTools.LoadSDF(path)

    # Get the name of the dependent column as the remaining one
    dependent_col = list(set(dataset.columns) - set(['ID', 'ROMol']))[0]

    smiles = [Chem.MolToSmiles(mol) for mol in dataset['ROMol']]
    dataset['SMILES'] = smiles

    # Drop ROMol column to standarize all datasets from different sources
    dataset = dataset.drop(['ROMol'], axis='columns')

    column_names = {
        'ID': 'CAS',
        'SMILES': 'SMILES',
        '%s' % dependent_col: 'Dependent',
    }

    # Rename the dataset with the required column names
    dataset.rename(index=str, columns=column_names, inplace=True)

    return dataset


def load_raw_list(path, ftype):
    '''Creates a dataframe from a file containing a list of entries (CAS/SMILES)
    '''
    with open(path, 'r') as input_file:
        content = input_file.read().split('\n')
        content = [clean_string(line) for line in content]
        content = list(filter(None, content))

        if ftype == 'smi':
            return pd.DataFrame({'SMILES': content})

        elif ftype == 'cas':
            # Eliminate entries not matching a date-like/CAS format
            content = [check_cas_format(cas) for cas in content]
            content = list(filter(None, content))
            return pd.DataFrame({'CAS': content,
                                 'SMILES': cas_to_smiles(content)})


def load_user_input(input_data):
    '''Loads and processes user's manual input.
    '''
    input_data = input_data.split(',')

    dataset = pd.DataFrame()

    for entry in input_data:
        entry = clean_string(entry)

        # Eliminate entries not matching a date-like/CAS format
        cas = check_cas_format(entry)
        if cas:
            dataset = dataset.append({'CAS': cas,
                                      'SMILES': cas_to_smiles(cas)},
                                      ignore_index=True)
        else:
            dataset = dataset.append({'SMILES': entry}, ignore_index=True)

    return dataset
