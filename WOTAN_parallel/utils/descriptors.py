# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools
# from mordred import Calculator, descriptors

from utils.web_scraping import cas_to_smiles


def calculate_descriptors(input_data, model_dict, acc_ftypes, model=True):
    '''Calculates Mordred molecular descriptors from SMILES/SDF inputs.
    Source: https://github.com/mordred-descriptor/mordred

    Keyword arguments:
        input_data (string) -- location of the input file (relative/absolute)
            or SMILES/CAS string
        acc_ftypes (list of tuples) -- accepted ftypes in app.py
        out_path (string) -- location of the output file (relative/absolute).
            If None, returns the descriptors dataframe
        model (bool) -- Indicates if there is already a QSAR model (for the App)
            If True, all the column elimination for being constant or correlated
            with other columns is not done because it is nonsense when there is
            already a model.

    Output:
        if out_path: None -- saves to an excel file a dataframe containing CAS,
            SMILES, dependent variable values and Mordred descriptors

        else: desc_df (dataframe) -- dataframe containing CAS, SMILES,
            dependent variable values and Mordred descriptors
    '''
    input_data, ftype = check_ftype(input_data)

    print('[+] Load input data and create raw dataset')
    # Get ignore3D and raw dataset
    ignore3D, dataset = load_dataframe(input_data, ftype, acc_ftypes)

    print('[+] Preprocess raw dataset')
    dataset = clean_raw_dataframe(dataset)

    # TODO: Convert SDF to SMILES and add to dataset
    print('[+] Get structures using RDKit')
    if ftype == 'sdf':
        # WARNING: load_dataframe option 'sdf' is not used
        # TODO: choose between 'load_dataframe' and 'SDMolSupplier'
        molec_struc = Chem.SDMolSupplier(input_data)
    else:
        molec_struc = [Chem.MolFromSmiles(smi) for smi in dataset['SMILES']]

    # Some CAS entries do not contain SMILES in the CACTUS server
    molec_struc, molec_struc_none = check_molec_struc(molec_struc)

    print('[+] Calculate Mordred descriptors')
    # calc = Calculator(descriptors, ignore_3D=ignore3D)
    # desc_df = calc.pandas(molec_struc, nproc=1)
    desc_df = calculate_all(molec_struc, model_dict)

    print('[+] Clean descriptors from missing, infinite and boolean values')
    desc_df = clean_decriptors(desc_df)

    if not model:
        print('[+] Drop colums with constant or near-constant values')
        desc_df = drop_constant_columns(desc_df)

        print('[+] Drop columns correlated between them')
        desc_df = drop_columns_correlated(desc_df)

        # TODO: Drop columns not correlated with the dependent variable

    print('[+] Insert data for CAS without SMILES (no descriptors available)')
    # TODO: Check correct inserts
    df_insert = pd.DataFrame(data={col: np.nan
                             for col in desc_df.columns.values},
                             index=[0])
    for idx in molec_struc_none:
        desc_df = insert_row(idx, desc_df, df_insert)


    # Insert from original dataset: Chemical name, CAS, SMILES, Dependent
    cols = dataset.columns.values
    for i in range(len(cols)):
        desc_df.insert(loc=i, column=cols[i], value=dataset[cols[i]].values)

    print('[*] Calculation of descriptors successfully finished\n\n')

    return desc_df


def check_cas_format(string):
    '''Checks and extracts the given string if it's in a date-like/CAS format.
    '''
    match = re.search(r'\d+-\d+-\d+', string)

    if match:
        return match.group()


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


def check_molec_struc(molec_struc):
    '''Checks if all the molecules structures have been obtained.
    '''
    molec_struc_none = [i for i in range(len(molec_struc))
                        if molec_struc[i] == None]
    molec_struc = list(filter(None, molec_struc))

    return molec_struc, molec_struc_none


def clean_decriptors(df, keep_na=True):
    '''Cleans the generated Mordred descriptors of Mordred's Missing errors,
    infinite values and NaN.

    Keyword arguments:
        df (dataframe) -- raw dataframe
        keep_na (bool) -- allows to keep o drop columns with NaN

    Output:
        clean_df (dataframe) -- dataframe with the NaN replaced or eliminated
    '''
    # Some values are set as mordred.error.Missing: must be converted into 0.0
    clean_df = df.apply(pd.to_numeric, errors='coerce')

    # Some values are infinite
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)

    # If used in APP, do not eliminate: all results even if NA are required.
    # If used in Pipeline, eliminate: NA interfere with the process.
    if not keep_na:
        clean_df = clean_df.dropna(axis='columns')

    # Convert True and False into 1 and 0
    clean_df = clean_df.astype(float)

    return clean_df


def clean_raw_dataframe(df, chars=['\n']):
    '''Cleans the raw dataframe from CAS duplicates, multiple CAS numbers in the
    same entry and non-desired characters (whitespaces, returns, etc.)
    '''
    clean_df = df.astype(str)

    # Force dataset dataframe to have all columns in standard_cols
    clean_df = fill_dataframe(clean_df)

    clean_df = clean_df.drop_duplicates(subset=['CAS'], keep='first')

    clean_df = split_character_in_column(clean_df,
                                          columns=['CAS'],
                                          char=['/', ' or '])

    for col in clean_df.columns:
        # Use strip to avoid Chemical name misspelling
        clean_df[col] = [clean_string(entry.strip(), chars)
                          for entry in clean_df[col]]

    return clean_df


def clean_string(string, chars=[' ', '\n']):
    '''Clean string from the desired characters.
    '''
    for ch in chars:
        string = string.replace(ch, '')

    return string


def fill_dataframe(df, cols=['Chemical name', 'CAS', 'SMILES', 'Dependent']):
    '''Fills a dataframe with 'Unknown' values in the specified columns in order
    to match a desired schema.
    '''
    # Dependent variable column is unknown since we are predicting
    for i in range(len(cols)):
        if cols[i] not in df.columns:
            if cols[i] == 'CAS':
                values = ['%s_%s' % (cols[i], n+1) for n in range(df.shape[0])]
            else:
                values = ['Unknown']*df.shape[0]

            df.insert(loc=i, column=cols[i], value=values)

    # Re-order column names
    df = df.reindex(cols, axis=1)

    return df


def insert_row(idx, df, df_insert):
    '''Auxiliary function for inserting a row.

    Keyword arguments:
        idx (int) -- index where to insert the row
        df (dataframe) -- raw dataframe
        df_insert (dataframe) -- row to be inserted

    Output:
        df (dataframe) -- dataframe with the inserted row
    '''
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]

    df = dfA.append(df_insert).append(dfB).reset_index(drop = True)

    return df


def load_dataframe(input_data, ftype, accepted_ftypes):
    '''Loads a dataframe from .xlsx, .csv, .txt , .smi or .cas files.
    '''
    if ftype:
        # Check if ftype is in the accepted ftypes in app.py
        accepted_ftypes = [tupla[1].replace('*.', '')
                           for tupla in accepted_ftypes]
        if ftype not in accepted_ftypes:
            print('[ERROR] File format is not accepted'), exit()

    if ftype == 'xlsx':
        return True, pd.read_excel(input_data, header=0)

    elif ftype == 'csv':
        return True, pd.read_csv(input_data, sep=',', header=0)

    elif ftype == 'txt':
        return True, pd.read_csv(input_data, sep='\t', header=0)

    elif ftype == 'sdf':
        return False, load_sdf(input_data)

    elif ftype == 'smi' or ftype == 'cas':
        return True, load_raw_list(input_data, ftype)

    elif not ftype:
        return True, load_user_input(input_data)


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


def split_character_in_column(df, columns=[], char=['\n'], pos=0):
    '''Splits column entries according to the characters present in them and
    replaces the entries by one of the resulting items.

    Keyword arguments:
        df (dataframe) -- dataframe to be processed
        columns (list) -- list of columns to be processed
        char (list) -- list of strings to be replaced
        pos (int) -- position to keep after spliting by the selected character

    Output:
        aux_df (dataframe) -- dataframe its column entries clean
    '''
    aux_df = df.copy() # Needed to bypass warnings of Panda's loc

    for col in columns:
        for ch in char:
            aux_df.loc[:, col] = [item.split(ch)[pos] for item in aux_df[col]]

    return aux_df
