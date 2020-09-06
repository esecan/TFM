

from mordred import Calculator
from mordred import BaryszMatrix

from rdkit import Chem
from rdkit.Chem import rdmolops

import numpy as np
import pandas as pd


def LP1(mol, **kwargs):
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    eigenvals = np.linalg.eigvalsh(adj_matrix)

    return eigenvals[-1]


def Barysz(mol, **kwargs):

    calc = Calculator(
    BaryszMatrix.BaryszMatrix(kwargs['prop'], kwargs['type'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    if type(desc.iloc[0, 0]) is np.float64:
        return(desc.iloc[0, 0])

    else: return 0


def Eigenvalues_all(mol):

    properties = ['Z', 'm', 'v', 'se', 'pe', 'are', 'p', 'i']
    types = ['SpAbs', 'SpMax', 'SpDiam', 'SpAD', 'SpMAD', 'LogEE', 'SM1', 'VE1', 'VE2', 'VE3', 'VR1', 'VR2', 'VR3']

    desc_df = pd.DataFrame()

    desc_name = 'LP1'
    desc_value = LP1(mol)
    desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

    desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    for type in types:
        for prop in properties:
            kwargs = {'prop': prop, 'type': type}

            desc_name = '{}_{}'.format(type, prop)
            desc_value = Barysz(mol, **kwargs)
            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
