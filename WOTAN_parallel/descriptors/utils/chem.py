import re
import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    rdmolops, Atom, PeriodicTable, Graphs, rdchem, rdMolDescriptors
)

from mordred._atomic_property import *

from networkx import Graph, floyd_warshall_numpy

from math import pi

from utils.json_files import load_json

CONFIG = load_json('descriptors/config/descriptors.json')


def load_descriptors(path):
    '''Loads and substitutes extrange characters in python in order to
    call functions
    '''
    with open(path, 'r') as infile:
        content = infile.read().split('\n')
        content = [re.sub('[\[\]-]', '', line) for line in content]

    return list(filter(None, content))


def replace_atoms_two_letters(smiles, letter='Q'):
    for atom in CONFIG['TWO_LETTERS']:
        smiles = smiles.replace(atom, letter)

    return smiles


def get_atoms(smiles):
    smiles = replace_atoms_two_letters(smiles)
    return re.sub(r'[^a-zA-Z]+', '', smiles)


def count_cycles(smiles, length):
    '''Counts the numbers of cycles and their length
    '''
    numbers = set([int(s) for s in smiles if s.isdigit()])

    # Get the cycles between numbers
    cycles = [re.search('%s(.*)%s' % (n, n) , smiles).group(1)
              for n in numbers]

    # Fix errors while dealing with atoms represented by more than one letter
    cycles = [replace_atoms_two_letters(cyc) for cyc in cycles]

    # Add one in length for the first atom before the first number
    cycles_len = [len(cyc) + 1 for cyc in cycles]

    # Filter those lengths according to the descriptor's definition
    cycles_len = [cyc_len for cyc_len in cycles_len
                  if cyc_len == length]

    return len(cycles_len)


def GetPrincipleQuantumNumber(atNum):
    '''Get principal quantum number for atom number
    '''
    if atNum <= 2:
        return 1
    elif atNum <= 10:
        return 2
    elif atNum <= 18:
        return 3
    elif atNum <= 36:
        return 4
    elif atNum <= 54:
        return 5
    elif atNum <= 86:
        return 6
    else:
        return 7


def get_atom_radii(atom_list):
    radii = list()
    for atom in atom_list:
        symbol = atom.GetSymbol()
        valence = atom.GetDegree()

        radius_key = '{}{}'.format(symbol, valence)
        radii.append(CONFIG['radii'][radius_key])

    return radii


def volume_vdw(vdw_rad):
    vdw_vol = (4/3)*pi*vdw_rad**3
    return vdw_vol


def laplacian_eigenvalues(mol, nSK, nBO):

    # Get the vertex degree matrix
    bond_list = []
    for bond in range(nBO):
        bond_list.append(mol.GetBondWithIdx(bond).GetBeginAtomIdx())
        bond_list.append(mol.GetBondWithIdx(bond).GetEndAtomIdx())

    vertex_matrix = np.zeros(shape=(nSK, nSK), dtype=np.object)
    i = 0
    for i in range(nSK):
        vertex_matrix[i,i] = bond_list.count(i)

    # Get the adjacency matrix and calculate the laplace matrix
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    laplace = np.subtract(vertex_matrix, adj_matrix)

    # Laplace matrix has to be converted into a array-list to get eigenvals
    eigenval = np.linalg.eigvalsh(laplace.tolist())

    return eigenval


def detour_matrix(mol):
    '''The detour matrix (square symmetric matrix representing a H-depleted
       molecular graph, whose entry i-j is the length of the longest path from
       vertex vi to vertex vj) is used to compute three descriptors.
    '''

    nSK = mol.GetNumAtoms()
    detour = np.zeros(shape=(nSK, nSK))

    # Calculate the detour matrix
    for distance in reversed(range(1, nSK)):
        for path in Chem.FindAllPathsOfLengthN(mol, distance+1, useBonds=0):
            # Sometimes it finds paths that cross an atom twice, so we want to
            # avoid this possibility checking if the last index is already in
            # the list of indexes of the path
            if [i for i in path].count(path[-1]) != 1:
                continue

            if detour[path[0], path[-1]] == 0 and path[0] != path[-1]:
                detour[path[0], path[-1]] = distance
                detour[path[-1], path[0]] = distance

    return detour


def GetWeightedMatrix(mol, **kwargs):
    '''The Wiener-type indices from weighted distance matrices (Whetw) are
       calculated by using the same formula as the Wiener index W applied to
       each weighted distance matrix, i.e. half-sum of matrix entries
    '''
    prop_fn_dict = {
        'Z': lambda atom: atom.GetAtomicNum(),
        'm': lambda atom: atom.GetMass(),
        'v': get_vdw_volume,
        'e': get_sanderson_en,
        'p': get_polarizability,
    }
    prop_fn = prop_fn_dict[kwargs['weight']]


    # Helper functions
    get_weight = lambda atom_i, atom_j, pi, fn: \
        (C * C) / (fn(atom_i) * fn(atom_j) * pi)

    fill_diagonal = lambda sp, mol, fn: \
        np.fill_diagonal(sp, [1. - C / fn(a) for a in mol.GetAtoms()])


    # Initialize carbon property value
    C = prop_fn(Chem.Atom(6))

    G = Graph()

    G.add_nodes_from(a.GetIdx() for a in mol.GetAtoms())


    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        atom_i = bond.GetBeginAtom()
        atom_j = bond.GetEndAtom()

        pi = bond.GetBondTypeAsDouble()

        w = get_weight(atom_i, atom_j, pi, prop_fn)

        G.add_edge(i, j, weight=w)


    sp = floyd_warshall_numpy(G)

    fill_diagonal(sp, mol, prop_fn)

    return sp


def BalabanJ(mol, dMat):
    # WARNING: adapted from RDkit
    '''Calculate Balaban's J value for a molecule

        **Arguments**
            - mol: a molecule
            - dMat: (optional) a distance/adjacency matrix for the molecule, if this
            is not provide, one will be calculated
            - forceDMat: (optional) if this is set, the distance/adjacency matrix
            will be recalculated regardless of whether or not _dMat_ is provided
            or the molecule already has one

        **Returns**
            - a float containing the J value

    We follow the notation of Balaban's paper:
    Chem. Phys. Lett. vol 89, 399-404, (1982)
    '''
    adjMat = Chem.GetAdjacencyMatrix(
        mol, useBO=0, emptyVal=0, force=0, prefix='NoBO'
    )

    s = sum(dMat).tolist()[0]
    q = mol.GetNumBonds()
    n = mol.GetNumAtoms()
    mu = q - n + 1

    suma = 0.
    nS = len(s)
    for i in range(nS):
        si = s[i]
        for j in range(i, nS):
            if adjMat[i, j] == 1:
                suma += 1. / float(np.sqrt(si * s[j]))

    if mu + 1 != 0:
        J = float(q) / float(mu + 1) * suma
    else:
        J = 0

    return J

def ValenceVertexDegree(atom):
    "Computes the valence vertex degree of an atom"

    qn = GetPrincipleQuantumNumber(atom.GetAtomicNum())

    if atom.GetAtomicNum() == 7 and atom.GetIsAromatic():
        Hs = rdchem.Atom.GetNumExplicitHs(atom)

    else:
        Hs = rdchem.Atom.GetNumImplicitHs(atom)

    #Dictionary to relate the qn with the value to compute the valence vertex degree
    qn_dict = {2:2, 3:10, 4:28, 5:46, 6:64}

    return atom.GetAtomicNum() - qn_dict[qn] - Hs
