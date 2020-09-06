# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math

from joblib import load

from sklearn import model_selection

import lightgbm as lgb

from utils.descriptors import calculate_descriptors
from utils.json_files import load_json

CONFIG = load_json('config/models_result_info.json')


def predict_qsar(data_input, selected_model, accepted_ftypes, model_name):
    '''Predicts the query molecule property from a QSAR model.

    Keyword arguments:
        selected_model (dict) -- descriptors and coeff of the selected model
        data_input (string) -- path to file/SMILES string
        accepted_ftypes (list of tuples) -- accepted ftypes in app.py

    Output:
        results_dict (dict) -- contains model name and parameters, as well as
            the results_df that is going to be displayed
    '''
    # Calculate the molecules descriptors
    desc_df = calculate_descriptors(data_input, selected_model, accepted_ftypes)

    # Round all float to 2 decimal positions
    desc_df = desc_df.round(2)

    # Fill NaN after rounding, round doesn't work in mixed type columns
    desc_df = desc_df.fillna(-999)

    print('[+] Model name: ', model_name)

    # Iterate over the different molecules in the descriptors dataframe

    # To predict with the models generated in pipeline
    # else:
    import pickle

    filename = 'sav_models/{}.sav'.format(model_name)

    loaded_model = pickle.load(open(filename, 'rb'))

    result = loaded_model.predict(desc_df.iloc[:,4:])

    results_info = []

    for i in range(len(result)):

        try:
            results_info.append(CONFIG[model_name][str(result[int(i)])])
        except:
            value_units = str(round(result[i], 2)) + ' ' + str(CONFIG[model_name]['Units'])
            results_info.append(value_units)

    results_df = desc_df.loc[:, :'SMILES']

    results_df.insert(loc=len(results_df.columns.values),
                      column='Prediction',
                      value=results_info)

    # Store results in a dictionary in case model name or parameters are needed
    results_dict = {}
    results_dict['model_parameters'] = selected_model
    results_dict['dataframe'] = results_df
    results_dict['descriptors'] = desc_df

    return results_dict
