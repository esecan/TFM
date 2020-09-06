# -*- coding: utf-8 -*-

import os

import numpy as np
from pandas import read_csv


def check_output_directory(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def data_from_csv(data_path, initial=0, column=-1, sep=',', header=None):
    ### DEPRECATED ###
    if data_path.endswith('.txt'): sep = '\t'
    
    data = read_csv(data_path, sep=sep, header=header)
    if type(column) == str:
        return np.array(data[column]), data.columns.values
    elif type(column) == int:
        return np.array(data.iloc[:, initial:column]), data.columns.values