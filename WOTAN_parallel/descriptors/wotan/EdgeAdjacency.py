import numpy as np
import pandas as pd

from rdkit import Chem

from math import log

from utils.json_files import load_json


CONFIG = load_json('descriptors/config/descriptors.json')


def EdgeAdjacencyIndex(mol, **kwargs):
    '''Molecular descriptors calculated from the edge adjacency matrix of
    a molecule

    Keyword arguments:
        type (string) -- either:
            'conn': edge connectivity index
            'eigen': eigenvalues from edge adjancency matrix
            'spectral': spectral moment from edge adjacency matrix

        weight (string) -- property to weight on:
            'none': no weight
            'degrees': edge degrees
            'dipole': dipole moments
            'resonance': resonance integrals

        order (int) -- order of any of the three types
            [0, 1] for edge connectivity indices
            [1, 15] for eigenvalues and spectral moments
    '''
    if kwargs['type'] == 'conn':
        raise NotImplementedError

    bond_number = mol.GetNumBonds()
    bond_list = []
    for bond in range(bond_number):
        bond_list.append((
            mol.GetBondWithIdx(bond).GetBeginAtomIdx(),
            mol.GetBondWithIdx(bond).GetEndAtomIdx()
        ))

    # WARNING: keep type of matrix as np.object. Otherwise, if the size of int
    # is reduced (such as np.int8), the exponentation will yield negative values
    # See: https://stackoverflow.com/questions/39602404/numpy-matrix-exponentiation-gives-negative-value
    edge_adj_matrix = np.zeros(shape=(bond_number, bond_number), dtype=np.object)

    # Fill the edge adjacency matrix
    for i in range(bond_number):
        a1, a2 = (
            mol.GetBondWithIdx(i).GetBeginAtomIdx(),
            mol.GetBondWithIdx(i).GetEndAtomIdx()
        )

        # Get the other pair of atoms in order to detect bond/edge adjancency
        for j in range(bond_number):
            a3, a4 = (
                mol.GetBondWithIdx(j).GetBeginAtomIdx(),
                mol.GetBondWithIdx(j).GetEndAtomIdx()
            )

            # If bonds next to each other, one atom is the same in both bonds
            if (a1 == a3 or a1 == a4 or a2 == a3 or a2 == a4) and i != j:
                edge_adj_matrix[i][j] = 1
                edge_adj_matrix[j][i] = 1

    # Weight the edge adjacency matrix dependending on the selected property
    # NOTE: for 'degree', the sum of the rows of the edge adjacency matrix is
    # performed, which requires the matrix to be filled in the previous loop
    if kwargs['weight'] != 'none':
        # TODO: check correct SMARTS is retrieved from CONFIG
        if kwargs['weight'] != 'degree':
            idx_dict = dict()
            for key in CONFIG[kwargs['weight']].keys():
                smarts = Chem.MolFromSmarts(key)
                idx_dict[key] = sorted(mol.GetSubstructMatches(smarts))

        for i in range(bond_number):
            if kwargs['weight'] == 'degree':
                edge_adj_matrix[i][i] = np.sum(edge_adj_matrix[:, i])

            # For 'dipole' and 'resonance'
            else:
                a1, a2 = (
                    mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                    mol.GetBondWithIdx(i).GetEndAtomIdx()
                )
                for key in idx_dict.keys():
                    if (a1, a2) in idx_dict[key]:
                        # Assume both atoms are the first and last atoms in key
                        edge_adj_matrix[i][i] = CONFIG[kwargs['weight']][key]


    if kwargs['type'] == 'spectral':
        matrix_order = np.linalg.matrix_power(edge_adj_matrix, kwargs['order'])
        trace = np.trace(matrix_order)
        result = log(1 + trace)

    if kwargs['type'] == 'eigen':
        if not edge_adj_matrix.any(): return 0

        eigenvals = np.linalg.eigvalsh(edge_adj_matrix.tolist())
        if len(eigenvals) < kwargs['order']: return 0

        result = eigenvals[-(kwargs['order'])]

    return result


def EdgeAdjacency_all(mol):

    desc_df = pd.DataFrame()

    dragon_name = {'spectral': 'ESpm', 'eigen': 'EEig'}
    weight_keys = {'none': 'u', 'degree': 'x', 'dipole': 'd', 'resonance': 'r'}

    for type in ['spectral', 'eigen']:
        for weight in ['none', 'degree', 'dipole', 'resonance']:
            for order in range(1, 16):
                kwargs = {'type': type, 'order': order, 'weight': weight}
                desc_name = '{}{:02d}{}'.format(
                    dragon_name[type], order, weight_keys[weight]
                )
                desc_value = EdgeAdjacencyIndex(mol, **kwargs)
                desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

                desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
