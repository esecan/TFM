# TODO: FINISH
# TODO: create general function

from rdkit import Chem

from itertools import chain

from utils.json_files import load_json

import pandas as pd


ATOM_CENTRED_SMARTS = load_json('descriptors/config/atom_centred.json')



def atom_centred(mol, **kwargs):
    '''Simple molecular descriptors defined as the number of specific atom
    types in a molecule. They are calculated by knowing the molecular
    composition and atom connectivities.
    '''
    # Add Hs for H-046 to H-055
    mol = Chem.AddHs(mol)

    # Get specified descriptor
    smarts = \
        [Chem.MolFromSmarts(sm) for sm in ATOM_CENTRED_SMARTS[kwargs['key']]]

    return sum([len(list(chain(mol.GetSubstructMatches(sm)))) for sm in smarts])


def AtomCentred_all(mol):

    desc_df = pd.DataFrame()

    for key in ATOM_CENTRED_SMARTS:

        kwargs = {'key': key}
        desc_name = key
        desc_value = atom_centred(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
