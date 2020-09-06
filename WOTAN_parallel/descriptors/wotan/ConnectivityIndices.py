import numpy as np

from rdkit import Chem
from rdkit.Chem import GraphDescriptors

from descriptors.utils.chem import GetPrincipleQuantumNumber

import pandas as pd


def connectivity_index(mol, **kwargs):
    '''Connectivity index calculated from a H-depleted molecular graph.

    Note: connectivity index with no weighting corresponds to Euclidean
    connectivity index in Todeschini, 2009.
    '''

    # TODO: Add tuples to compute the whole set of conenctivity descriptors

    order = kwargs['order']

    # Define degree
    get_degree = lambda atom: atom.GetDegree()


    # For the solvation connectivity index the order should be increased by one,
    # and the fluors are considered as hydrogens
    if kwargs['type'] == 'sol':
        # Update order
        if order > 0: order += 1

        # Fluorine atoms are not included in the graph, their dimension
        # being very close to that of the hydrogen atom
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 9:
                emol = Chem.EditableMol(mol)
                emol.ReplaceAtom(atom.GetIdx(), Chem.Atom(1))

                mol = emol.GetMol()
                Chem.SanitizeMol(mol)

    mol = Chem.RemoveHs(mol)

    result = []

    # Depending on the order the way to compute the index is different
    if order == 0 and kwargs['type'] != 'v':
        result = [get_degree(x) for x in mol.GetAtoms()]
        # Eliminate zeros
        result = np.array(result)[np.nonzero(result)]
        result = [1/np.sqrt(res) for res in result]

    elif order == 1 and kwargs['type'] != 'v':
        result = [get_degree(x.GetBeginAtom()) * get_degree(x.GetEndAtom())
                 for x in mol.GetBonds()]
        result = np.array(result)[np.nonzero(result)]
        result = [1/np.sqrt(res) for res in result]

    elif order > 1 and kwargs['type'] != 'v':
        if kwargs['type'] == 'sol':
            for path in Chem.FindAllPathsOfLengthN(mol, order, useBonds=0):
                delta_prod = qn_prod = 1.0
                for idx in path:
                    atom = mol.GetAtomWithIdx(idx)
                    delta = get_degree(atom)

                    delta_prod *= delta

                    if not delta: return 0.0

                    qn = GetPrincipleQuantumNumber(atom.GetAtomicNum())
                    qn_prod *= qn

                result.append(qn_prod / np.sqrt(delta_prod))



        elif kwargs['type'] == '':
            for path in Chem.FindAllPathsOfLengthN(mol, order+1, useBonds=0):
                delta_prod = qn_prod = 1.0
                for idx in path:
                    atom = mol.GetAtomWithIdx(idx)
                    delta = get_degree(atom)

                    delta_prod *= delta

                    if not delta: return 0.0

                result.append(1 / np.sqrt(delta_prod))


    # If valence type we use rdkit implementation
    elif kwargs['type'] == 'v':
        desc_fn_name = 'Chi{}{}'.format(kwargs['order'], kwargs['type'])
        desc_fn = getattr(Chem.GraphDescriptors, desc_fn_name)
        result = desc_fn(mol)

    # Negative orders
    else:
        raise NotImplementedError

    # Get the final result depending on the type
    if kwargs['type'] == 'sol':
        result = 1/(2**(order))*sum(result)

    elif kwargs['type'] == '':
        result = sum(result)

    # Average if required
    if kwargs['avg']:
        if order >= 1:
            n_path = len(Chem.FindAllPathsOfLengthN(mol, order))

        # TODO: Check what happens when order == 0
        else:
            n_path = mol.GetNumAtoms()

        # Avoid division by zero errors
        if n_path: result /= n_path
        else:
            return 0.0

    return result

def ConnectivityIndices_all(mol):
    order_values = [0, 1, 2, 3 , 4, 5]
    types = ['', 'v', 'sol']
    average =  ['', 'A']

    desc_df = pd.DataFrame()

    for order in order_values:
        for type in types:
            for avg in average:
                try:
                    kwargs = {'order': order, 'type': type, 'avg': avg}
                    desc_name = 'X{}{}{}'.format(order, type, avg)
                    desc_value = connectivity_index(mol, **kwargs)
                    desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

                    desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

                except: continue

    return desc_df
