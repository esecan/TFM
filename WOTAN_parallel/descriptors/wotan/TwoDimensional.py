import pandas as pd

from itertools import combinations_with_replacement

from rdkit import Chem
from rdkit.Chem import rdmolops


def topological_distance(mol, **kwargs):
    '''Presence/absence of [atom] - [atom] at topological distance [distance]

    Keyword arguments:
        atoms (tuple of ints) -- atomic numbers to be found
        distance -- distance between atoms
        type: binary (B[...]), frequency (F[...]), sum (T(...))
    '''
    # WARNING: All the dragon test compounds are 0 for some descriptors
    mol = Chem.RemoveHs(mol)

    # TODO: Check case sensitive cases
    atom_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # If both atoms are not in the molecule
    if not set(kwargs['atoms']).issubset(atom_list):
        return 0

    dm = rdmolops.GetDistanceMatrix(mol)
    df = pd.DataFrame(dm, columns=atom_list, index=atom_list)

    # Iterate distance matrix over columns
    found = []

    for i in range(dm.shape[1]):
        query_atom = atom_list[i]

        # Force to visit only once with just one condition (avoid the
        # condition of "query_atom == kwargs['atoms'][1]"). Otherwise, it
        # would be found twice, since both atoms are being queried

        if query_atom == kwargs['atoms'][0]:
            for j in range(dm.shape[1]):
                second_atom = atom_list[j]

                if second_atom == kwargs['atoms'][1]:
                    if dm[i, j] == kwargs['distance']:
                        found.append(dm[i, j])


    if   kwargs['type'] == 'binary':    return int(sum(found) > 0)
    elif kwargs['type'] == 'frequency': return len(found)
    elif kwargs['type'] == 'sum':       return sum(found)


def TwoDimensional_all(mol):
    heteroatoms = [7, 8, 16, 15, 9, 17, 35, 53]
    all_atoms = [6, 5, 14] + heteroatoms

    hetero_combs = list(combinations_with_replacement(heteroatoms, 2))
    atom_combs = list(combinations_with_replacement(all_atoms, 2))

    distances = [i for i in range(1, 11)]
    types = ['binary', 'frequency']

    desc_df = pd.DataFrame()

    # TODO: sort descriptors names/columns??

    get_symbol = lambda z: Chem.Atom(z).GetSymbol()

    for distance in distances:
        # For binary and frequency descriptors
        for atoms in atom_combs:
            atoms_symbols = list(map(get_symbol, atoms))
            for type in types:
                kwargs = {'atoms': atoms, 'distance': distance, 'type': type}
                desc_name = type[0].upper() + \
                    '{:02d}'.format(distance) + \
                    '[{}-{}]'.format(*atoms_symbols)
                desc_value = topological_distance(mol, **kwargs)
                desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

                desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

        # For sum of topological distance descriptors
        for atoms in hetero_combs:
            atoms_symbols = list(map(get_symbol, atoms))
            kwargs = {'atoms': atoms, 'distance': distance, 'type': 'sum'}
            desc_name = 'T' + \
                '{:02d}'.format(distance) + \
                '({}..{})'.format(*atoms_symbols)
            desc_value = topological_distance(mol, **kwargs)
            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
