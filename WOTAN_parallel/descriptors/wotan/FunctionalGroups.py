# TODO: FINISH
# TODO: create general function

from rdkit import Chem

from itertools import chain

from utils.json_files import load_json

import pandas as pd


FUNCTIONAL_GROUP_SMARTS = load_json('descriptors/config/functional_group.json')



def functional_group(mol, **kwargs):
    '''Simple molecular descriptors defined as the number of specific functional
    groups in a molecule. They are calculated by knowing the molecular
    composition and atom connectivities.
    '''
    # Add Hs for H-046 to H-055
    mol = Chem.AddHs(mol)

    # Get specified descriptor
    smarts = \
        [Chem.MolFromSmarts(sm) for sm in FUNCTIONAL_GROUP_SMARTS[kwargs['group']]]

    return sum([len(list(chain(mol.GetSubstructMatches(sm)))) for sm in smarts])


def FunctionalGroup_all(mol):

    desc_df = pd.DataFrame()

    for desc_name in FUNCTIONAL_GROUP_SMARTS.keys():
        kwargs = {'group': desc_name}
        desc_value = functional_group(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
