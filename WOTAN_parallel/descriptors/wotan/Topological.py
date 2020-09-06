import numpy as np
import collections
import itertools

import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    rdmolops,
    Atom,
    PeriodicTable,
    GraphDescriptors,
    Descriptors3D,
    rdMolDescriptors,
)

from scipy.stats.mstats import gmean

from mordred import WalkCount, PathCount
from mordred import Calculator
from mordred import BaryszMatrix
from mordred import ZagrebIndex
from mordred import DetourMatrix
from mordred import TopologicalIndex
from mordred import EccentricConnectivityIndex

from descriptors.utils.chem import *

from descriptors.wotan.TwoDimensional import topological_distance

from math import sqrt, log, log10, factorial

from utils.json_files import load_json


CONFIG = load_json('descriptors/config/descriptors.json')


def ZM(mol, **kwargs):
    '''The first Zagreb index (ZM1) is the sum of the square vertex degrees
    of all the non-hydrogen atoms.

    Mordred implementation values:
        - ZM1: (1, 1)
        - ZM2: (2, 1)
    '''

    # Manual implementation

    ZMV = 0
    degree = []

    if kwargs['type'] == 'v':
        if kwargs['order'] == 1:
            for atom in mol.GetAtoms():
                ZMV += ValenceVertexDegree(atom) ** 2

        elif kwargs['order'] == 2:
            for bond in mol.GetBonds():
                begin_val = ValenceVertexDegree(bond.GetBeginAtom())
                end_val = ValenceVertexDegree(bond.GetEndAtom())
                degree.append(begin_val * end_val)

            ZMV = sum(degree)

        return ZMV

    else:
        if kwargs['order'] == 1:
            for atom in mol.GetAtoms():
                degree.append(atom.GetDegree()**2)

        elif kwargs['order'] == 2:
            for bond in mol.GetBonds():
                begin_degree = bond.GetBeginAtom().GetDegree()
                end_degree = bond.GetEndAtom().GetDegree()
                degree.append(begin_degree * end_degree)

        return sum(degree)


def Qindex(mol, **kwargs):

    nSK = mol.GetNumAtoms()

    ZM1 = ZM(mol, type = '', order = 1)

    return 3 - 2 * nSK + ZM1/2


def Nar(mol, **kwargs):
    '''Returns the Narumi topological index, related to molecualr branching

    kwargs['type']
        - simple
        - harmonic
        - geometrical
    '''

    degree_list = []
    degree_product = 1
    degree_sum = 0

    for atom in mol.GetAtoms():
        degree_list.append(atom.GetDegree())

    if kwargs['type'] == 'S':
        for degree in degree_list:
            degree_product *= degree

        try: return log(degree_product)
        except: return 0

    elif kwargs['type'] == 'H':
        nSK = mol.GetNumAtoms()
        if sum(degree_list) > 0:
            for degree in degree_list:
                try:
                    degree_sum += degree ** -1
                except ZeroDivisionError:
                    continue

            return nSK/degree_sum

        else:
            return 0

    elif kwargs['type'] == 'G':
        if sum(degree_list) > 0:
            return gmean(degree_list)

        else:
            return 0


def Xt(mol, **kwargs):
    '''Returns the reciprocal square root of the Narumi simple topological
       index SNar
    '''
    try:
        Xt = 1 / sqrt(Nar(mol, type = 'S'))
        return Xt

    except: return 0


# TODO: Check Dz descriptor, GetNouterelectrons not found
def Dz(mol, **kwargs):
    '''Returns the Pogliani index, the sum over all non-hydrogen atoms of a
       modified vertex degree calculated as the ratio of the number of valence
       electrons over the principal quantum number of an atom
    '''
    sum_dz = 0
    PER_TABLE = Chem.rdchem.GetPeriodicTable()

    for atom in mol.GetAtoms():
        sum_dz += (
        PeriodicTable.GetNouterElecs(PER_TABLE, Atom.GetAtomicNum(atom))
        / GetPrincipleQuantumNumber(atom.GetAtomicNum())
        )

    return sum_dz

def Ram(mol, **kwargs):
    '''Returns the ramification index, calculated as the sum over all the
       vertex degrees greater than two of the vertex degree minus 2
    '''

    ram = 0

    for atom in mol.GetAtoms():
        degree_value = atom.GetDegree()
        if degree_value > 2:
            ram += degree_value -2

    return (max(0, ram))


def Pol(mol, **kwargs):
    '''It is calculated on the distance matrix as the number of pairs of vertices at
       a topological distance equal to three
    '''

    dm = rdmolops.GetDistanceMatrix(mol)
    unique, counts = np.unique(dm, return_counts=True)
    a = dict(zip(unique, counts))

    # The result is divided by zero because the distance matrix is simmetric
    # and we want the single occurrences
    try: return a[3.0] / 2

    except: return 0


def DistanceMatrix_row(mol, **kwargs):
    '''Returns the logarithm of the product of the distance degrees of all
       non-hydrogen atoms
    '''
    atom_dist_sum = 0
    atom_dist_product = 1
    nSK = mol.GetNumAtoms()

    dm = rdmolops.GetDistanceMatrix(mol)
    atom_distance_degrees = dm.sum(axis = 1)

    for atom in atom_distance_degrees:
        atom_dist_product *= atom
        atom_dist_sum += atom

    if kwargs['type'] == 'LPRS' and atom_dist_product > 0:
        return log(atom_dist_product)

    elif kwargs['type'] == 'VDA' and atom_dist_sum > 0:

        return atom_dist_sum / nSK

    elif kwargs['type'] == 'MDDD' and atom_dist_sum > 0:
        vda = atom_dist_sum / nSK
        return sum(abs(atom_distance_degrees - vda)) / nSK

    else:
        return 0


def MSD(mol, **kwargs):
    remover = Chem.SaltRemover.SaltRemover(defnData="[Na,Cl]")
    mol = remover.StripMol(mol)

    dm = rdmolops.GetDistanceMatrix(mol)
    nSK = mol.GetNumAtoms()

    # Dimension of matrix must be number of atoms x number of atoms
    assert dm.shape[0] == dm.shape[1]
    assert nSK == dm.shape[0]

    double_sum = 0
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            # Avoid when i == j (same atom)
            if i != j:
                double_sum += dm[i, j]**2

    try: return sqrt(double_sum) / (nSK*(nSK - 1))
    except: return 0


def SMTI(mol, **kwargs):
    '''The Schultz Molecular Topological Index (SMTI) is derived from the
       adjacency matrix A, the distance matrix D and the A-dimensional column
       vector v constituted by the vertex degree of the atoms in the H-depleted
       molecular graph
    '''
    degree = []

    dm = rdmolops.GetDistanceMatrix(mol)
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    for atom in mol.GetAtoms():
        if kwargs['type'] == 'V':
            degree.append(ValenceVertexDegree(atom))

        else:
            degree.append(atom.GetDegree())

    smti_sum = dm + adj_matrix
    smti = np.matmul(smti_sum, degree)
    return sum(smti)


def GMTI(mol, **kwargs):
    '''The Gutman Molecular Topological Index (GMTI) is calculated using ...
       where δ refers to vertex degrees, nSK to the number of non-hydrogen atoms
       and dij to the topological distance between two atoms

       The GMTIV index is obtained in the same way as the GMTI index using the
       valence vertex degree in place of the simple vertex degree.
    '''

    gmti = 0
    dm = rdmolops.GetDistanceMatrix(mol)

    for atom_a in mol.GetAtoms():
        idx_a = atom_a.GetIdx()
        for atom_b in mol.GetAtoms():
            idx_b = atom_b.GetIdx()
            if kwargs['type'] == 'V':
                gmti += (ValenceVertexDegree(atom_a)*ValenceVertexDegree(atom_b)
                        *dm[idx_a, idx_b])

            else:
                gmti += atom_a.GetDegree()*atom_b.GetDegree()*dm[idx_a, idx_b]

    return gmti / 2


def Xu(mol, **kwargs):
    '''Returns the Xu index_jhetz
    '''
    nSK = mol.GetNumAtoms()
    dm = rdmolops.GetDistanceMatrix(mol)
    atom_distance_degrees = dm.sum(axis = 1)
    sum_num = 0
    sum_denom = 0

    for atom in mol.GetAtoms():
        sum_num += atom.GetDegree()*(atom_distance_degrees[atom.GetIdx()]**2)
        sum_denom += atom.GetDegree()*atom_distance_degrees[atom.GetIdx()]

    if sum_denom > 0:
        xu = sqrt(nSK) * log(sum_num / sum_denom)
        return xu

    else:
        return 0


def SPI(mol, **kwargs):
    '''The superpendentic index (SPI) is calculated as the square root of the
       sum of the products of the nonzero row elements in a reduced distance
       matrix where the rows correspond to all non-hydrogen atoms and the
       columns to only the terminal atoms
    '''
    dm = rdmolops.GetDistanceMatrix(mol)
    terminal_atoms = []

    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1:
            terminal_atoms.append(atom.GetIdx())

    dm_reduced = dm[:,terminal_atoms]
    sum = 0

    for i in range(dm_reduced.shape[0]):
        for element in dm_reduced[i, :]:
            if element == 0: element = 1
            sum += log(element)

    return sqrt(sum)


def Wiener(mol, **kwargs):
    '''The Wiener index (W) is calculated as the half-sum of all topological
       distances collected in the distance matrix
    '''
    nSK = mol.GetNumAtoms()
    sum = 0
    dm = rdmolops.GetDistanceMatrix(mol)

    for i in range(dm.shape[0]):
        for element in dm[i, :]:
            sum += element

    # Mean wiener index
    if kwargs['type'] == 'A':
        if nSK > 1:
            return sum / ((nSK ** 2) - nSK)
        else:
            return 0

    # Wiener index
    else:
        return sum / 2


def Whetw(mol, **kwargs):
    '''The Wiener-type indices from weighted distance matrices (Whetw) are
       calculated by using the same formula as the Wiener index W applied to
       each weighted distance matrix, i.e. half-sum of matrix entries
    '''
    sp = GetWeightedMatrix(mol, **kwargs)

    return sum(sum(np.array(sp))) / 2


def Jhetw(mol, **kwargs):
    '''The Balaban-type indices from weighted distance matrices (Jhetw) are
    calculated by using the same formula as the Balaban distance connectivity
    index J applied to each weighted distance matrix.
    '''
    sp = GetWeightedMatrix(mol, **kwargs)

    # WARNING: not working while using RDkit implementation (maybe for using sp)
    # return GraphDescriptors.BalabanJ(mol, dMat=sp)

    return BalabanJ(mol, dMat=sp)


def J(mol, **kwargs):
    '''The Balaban distance connectivity index (J)is calculated using a Randic
    connectivity index-type formula where the vertex degrees are substituted by
    the distance degrees and a normalisation factor makes this index
    substantially independent of the molecule size and number of rings.
    '''
    dm = rdmolops.GetDistanceMatrix(mol)

    return GraphDescriptors.BalabanJ(mol, dMat=dm)


def Har(mol, **kwargs):
    '''The Harary H index (Har) is calculated as the sum of all the reciprocal
       topological distances in a H-depleted molecular graph
    '''
    dm = rdmolops.GetDistanceMatrix(mol)
    suma = 0
    for i in range(dm.shape[0]):
        for j in range(dm.shape[0]):
            if i != j:
                if kwargs['order'] == 1:
                    suma += 1 / dm[i, j]
                elif kwargs['order'] == 2:
                    suma += 1 / (dm[i, j] ** 2)

    return suma / 2


def QW(mol, **kwargs):
    '''The quasi-Wiener index (QW) is calculated as the product of the number
       of non-H atoms (nSK) and the sum of the reciprocal nSK – 1 positive
       eigenvalues of the Laplacian matrix
    '''
    nSK = mol.GetNumAtoms()
    nBO = mol.GetNumBonds()

    eigenval =  laplacian_eigenvalues(mol, nSK, nBO)


    # TODO: Check if it's working properly
    if len(eigenval) <= 1:
        return 0
    else:

        eigenval = sum(1/eigenval[i] for i in range(1, len(eigenval)) if eigenval[i] != 0)

    qw = nSK * eigenval

    return qw


def MoharIndex(mol, **kwargs):
    '''First or second mohar index

    Keyword arguments:
        type -- first or second
    '''
    nSK = mol.GetNumAtoms()
    nBO = mol.GetNumBonds()

    if kwargs['order'] == 1:

        if nBO == 0: return 0

        TI1 = 2 * log10(nBO / nSK) * QW(mol)

        return TI1

    if kwargs['order'] == 2:

        eigenval =  laplacian_eigenvalues(mol, nSK, nBO)

        # TODO: Check if it's working properly
        if len(eigenval) <= 1: return 0
        else:
            eigenval = eigenval[1]
            # for i in range(len(eigenval)):
            #     if eigenval[i] != 0:
            #         break
        # eigenval = eigenval[i]

        TI2 = 4/(nSK*eigenval)

        return TI2


def STN(mol, **kwargs):
    '''The spanning tree number (STN) is the product of the positive nSK – 1
       eigenvalues of the Laplacian matrix divided by the number of non-H atoms (nSK)
    '''
    nSK = mol.GetNumAtoms()
    nBO = mol.GetNumBonds()

    eigenval =  laplacian_eigenvalues(mol, nSK, nBO)
    product = 1

    for value in eigenval[1: ]:
        product *= value

    stn = product / nSK

    if product != 0:
        return round(log(stn), 3)

    else: return 0


def HyDp(mol, **kwargs):
    '''The hyper-distance-path index (HyDp) is calculated as the half-sum of
       the entries of the distance-path matrix
    '''
    dm = rdmolops.GetDistanceMatrix(mol)

    # To get the distance path matrix from the distance matrix
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            if kwargs['type'] == 'R' and dm[i, j] != 0:
                dm[i, j] = 1 / ((dm[i, j] ** 2 + dm[i, j]) / 2)

            else:
                dm[i, j] = (dm[i, j] ** 2 + dm[i, j]) / 2


    hydp = sum(dm.sum(axis = 1)) / 2

    return hydp


def DetourIndex(mol, **kwargs):
    '''The detour matrix (square symmetric matrix representing a H-depleted
       molecular graph, whose entry i-j is the length of the longest path from
       vertex vi to vertex vj) is used to compute three descriptors.

    w : calculated as the half-sum of the entries of the detour matrix

    '''

    # Calculate the detour index
    w = sum(sum(detour_matrix(mol))) / 2

    return w


def HyperDetour(mol, **kwargs):
    '''
kwargs[type] == ''
       ww: calculated as the half-sum of the entries of the detour-path matrix.

kwargs[type] == 'reciprocal'
       Rww: calculated as the half-sum of the reciprocal entries of the
            detour-path matrix
    '''
    nSK = mol.GetNumAtoms()
    detour = detour_matrix(mol)


    if kwargs['type'] == 'R':
        reciprocal_hyper_detour = np.zeros(shape=(nSK, nSK))
        for i in range(nSK):
            for j in range(nSK):
                if detour[i, j] != 0:
                    reciprocal_hyper_detour[i, j] = 1 / ((detour[i, j] ** 2 + detour[i, j]) / 2)

        hyper = sum(sum(reciprocal_hyper_detour)) / 2


    else:
        hyper_detour = np.zeros(shape=(nSK, nSK))
        for i in range(nSK):
            for j in range(nSK):
                if detour[i, j] != 0:
                    hyper_detour[i, j] = (detour[i, j] ** 2 + detour[i, j]) / 2

        hyper = sum(sum(hyper_detour)) / 2

    return hyper


def DistanceDetourIndex(mol, **kwargs):
    '''
       Calculated as the half-sum of the entries of the distance/detour
       quotient matrix. It was proposed as an index of molecular cyclicity,
       showing regular variation with increase in cyclicity in graphs of the
       same size.

    kwargs[length] = 3 - 12
       Distance/detour ring indices (D/Drk) are calculated by summing up
       distance/detour quotient matrix row sums of vertices belonging to single
       rings in the molecule
    '''
    nSK = mol.GetNumAtoms()
    dm = rdmolops.GetDistanceMatrix(mol)
    detour = detour_matrix(mol)
    distance_detour = np.zeros(shape=(nSK, nSK))

    for i in range(nSK):
        for j in range(nSK):
            if detour[i, j] != 0:
                distance_detour[i, j] = dm[i, j] / detour[i, j]

    if kwargs['length']:
        smart = '*1{}*1'.format('*' * (kwargs['length'] - 2 ))
        ring_smart = Chem.MolFromSmarts(smart)
        matches = mol.GetSubstructMatches(ring_smart)
        DD = 0
        for match in matches:
            for idx in match:
                DD += sum(distance_detour[idx])

    else:
        DD = sum(sum(distance_detour)) / 2

    return DD


def Wap(mol, **kwargs):
    '''The all-path Wiener index (Wap) is the half-sum of path degrees over
       all vertices in a H-depleted molecular graph, the path degree of a
       vertex being the sum of the lengths of all paths starting from the
       considered vertex
    '''
    nSK = mol.GetNumAtoms()
    wap = 0
    for distance in range(1, nSK):
        for path in Chem.FindAllPathsOfLengthN(mol, distance+1, useBonds=0):
            if [i for i in path].count(path[-1]) != 1:
                continue

            wap += distance

    return wap


def KierShapeIdx(mol, order, alpha=0.0):
    # Change N for Kier's alpha-modified shape indicess
    N = mol.GetNumAtoms() + alpha

    P = len(Chem.FindAllPathsOfLengthN(mol, order+1, useBonds=0))
    if not P:
        if len(mol.GetAtoms()) == 2: return 1.0
        else: return 0.0

    K = (N+1-order)*((N-order)**2) / ((P+alpha)**2)

    # For order 3 and N even
    if order == 3 and N % 2 == 0:
        K = (N-3)*((N-2)**2) / (P**2)

    return K


def SkK(mol, **kwargs):
    # Avoid confusion when getting degree/valence of an atom
    mol = Chem.AddHs(mol)

    # Heavy atoms as defined by Kier
    # [C, N, O, F, P, Cl, Br, I]
    # [6, 7, 8, 9, 15, 17, 35, 53]

    # Heavy atoms as defined in Dragon
    # [C, N, O, B, Al, Si, Fe, Co, P, S, F, Cl, Br, I, Ni, Cu, Zn, Sn, Gd]
    # [6, 7, 8, 5, 13, 14, 26, 27, 15, 16, 9, 17, 35, 53, 28, 29, 30, 50, 64]

    heavy_atoms = \
        [6, 7, 8, 5, 13, 14, 26, 27, 15, 16, 9, 17, 35, 53, 28, 29, 30, 50, 64]

    atom_list = [atom for atom in mol.GetAtoms()
                if Atom.GetAtomicNum(atom) in heavy_atoms]

    # Get atom radii
    try:
        radii = get_atom_radii(atom_list)

    except KeyError:
        return 0

    # Calculate alpha
    C_radius = CONFIG['radii']['C4']
    alpha = sum([((radius/C_radius) - 1) for radius in radii])

    # Avoid counting Hs
    mol = Chem.RemoveHs(mol)

    return KierShapeIdx(mol, kwargs['order'], alpha=alpha)


def Phi(mol, **kwargs):
    # The Kier flexibility index (PHI) is derived from the Kier alpha-modified
    # shape indices S1K and S2K
    nSK = mol.GetNumAtoms()

    return (SkK(mol, order = 1) * SkK(mol, order = 2)) / nSK


def Bli(mol, **kwargs):
    # The Kier benzene-likeliness index (BLI) is calculated by dividing the
    # first-order valence connectivity index X1V by the number of non-H bonds (nBO)
    # of the molecule and then normalising on the benzene molecule
    from descriptors.wotan.ConnectivityIndices import connectivity_index

    # Compute the
    benzene_mol = Chem.MolFromSmiles('C1=CC=CC=C1')

    benzene_bli = (
                  connectivity_index(benzene_mol, order = 1, type = 'v', avg = '') /
                  benzene_mol.GetNumBonds()
                  )

    nBO = mol.GetNumBonds()
    x1v = connectivity_index(mol, order = 1, type = 'v', avg = '')

    if nBO > 0:
        return (x1v / nBO) / benzene_bli

    else: return 0




def PWk(mol, **kwargs):

    nSK = mol.GetNumAtoms()
    if nSK > 1:
        # desc = 0.0
        # for i in range(kwargs['order']+1):
        PathC = Calculator(
            PathCount.PathCount(order=kwargs['order'], pi=False, total=False, log=False)
        ).pandas([mol], nproc=1, quiet=True).iloc[0,0]

        WalkC = Calculator(
            WalkCount.WalkCount(order=kwargs['order'], total=False, self_returning=False)
        ).pandas([mol], nproc=1, quiet=True).iloc[0,0]

        desc = PathC / WalkC

    else: return 0

    return desc / nSK


def PJI2(mol, **kwargs):
    '''The 2D Petitjean shape index (PJI2) is calculated as the difference
       between topological diameter and radius, then divided by the radius, the
       topological diameter being the maximum atom eccentricity and the radius
       the minimum atom eccentricity
    '''
    nSK = mol.GetNumAtoms()

    if nSK > 1:
        pji2 = (max(ECC(mol, type='')) - min(ECC(mol, type=''))) / min(ECC(mol, type=''))

    else: return 0

    return pji2


def ECC(mol, **kwargs):
    '''The eccentricity (ECC) is the sum over all non-hydrogen atoms of the atom
       eccentricity which is the maximum distance from an atom to any other atoms

       Kwargs['type']
            - eccentricity (ECC)
            - average (AECC)
            - eccentric (DECC)
            - index (CSI)
    '''
    nSK = mol.GetNumAtoms()
    dm = rdmolops.GetDistanceMatrix(mol)
    ecc = dm.max(axis=0)


    # Eccentricity ECC
    if kwargs['type'] == 'E':
        result = sum(ecc)

    # Average Eccentricity AECC
    elif kwargs['type'] == 'AE':
        result = sum(ecc) / nSK

    # Eccentric DECC
    elif kwargs['type'] == 'DE':
        avg = sum(ecc) / nSK

        result = 0
        for atom in mol.GetAtoms():
            result += abs(ecc[atom.GetIdx()] - avg)
        result /= nSK

    # Eccentricity connectivity index CSI
    elif kwargs['type'] == 'index':
        result = 0
        for atom in mol.GetAtoms():
            vertex_degree = atom.GetDegree()
            result += (vertex_degree * ecc[atom.GetIdx()])

    elif kwargs['type'] == 'radial':

        nSK = mol.GetNumAtoms()
        result = 0
        ecc_unique = list(set(ecc))

        for distance in ecc_unique:
            nk = ecc.tolist().count(distance)
            result -= (nk / nSK) * log((nk / nSK), 2)

    else:
        result = ecc

    return result


def intrinsic_state(atom):
    '''Intrinsic state of an atom (see Dragon for calculation)

    Requirements: H-depleted molecular structure
    '''
    # L: principal quantum number
    L = GetPrincipleQuantumNumber(atom.GetAtomicNum())

    # TODO: EACH ATOM A KEY IN GROUP NUMBER
    # delta_v: number of valence electrons (valence vertex degree)
    delta_v = ValenceVertexDegree(atom)

    # TODO: CHEEECK
    # delta: number of sigma electrons (vertex degree) of the ith atom
    delta = atom.GetDegree()

    try:
        return (((2/L)**2)*delta_v + 1) / delta
    except ZeroDivisionError:
        return 0


def delta_intrinsic_state(mol):
    '''(delta I) is the field effect on the ith atom due to the perturbation
    of all other atoms as defined by Kier and Hall
    '''
    atom_list = list(mol.GetAtoms())
    delta_I = np.zeros(shape=(len(atom_list), 1))

    dm = rdmolops.GetDistanceMatrix(mol)

    field_effect = lambda atom1, atom2, idx1, idx2: \
        (intrinsic_state(atom1) - intrinsic_state(atom2)) / \
        ((dm[idx1, idx2] + 1)**2)

    for atom1 in atom_list:
        i = atom_list.index(atom1)
        for atom2 in atom_list:
            j = atom_list.index(atom2)
            if i != j:
                delta_I[i] += field_effect(atom1, atom2, i, j)

    return delta_I


def DELS(mol, **kwargs):
    '''Sum of delta Ii (see Gramatica et al. (2000) for calculation)
    '''
    return np.abs(delta_intrinsic_state(mol)).sum()


def MAXDN(mol, **kwargs):
    '''Max of delta Ii < 0 (see Gramatica et al. (2000) for calculation)
    '''
    delta_I = delta_intrinsic_state(mol)

    try:
        return abs(np.amin(delta_I[delta_I < 0]))

    except ValueError:
        return 0


def MAXDP(mol, **kwargs):
    '''Max of delta Ii > 0 (see Gramatica et al. (2000) for calculation)
    '''
    delta_I = delta_intrinsic_state(mol)

    try:
        return abs(np.amax(delta_I[delta_I > 0]))

    except ValueError:
        return 0


def Unipolarity(mol, **kwargs):
    '''The unipolarity (UNIP) is the minimum value of the vertex distance degrees
    '''
    dm = rdmolops.GetDistanceMatrix(mol)
    atom_distance_degrees = dm.sum(axis = 1)

    return min(atom_distance_degrees)


def Centralization(mol, **kwargs):
    '''Obtained from the number of atoms (nSK) the wiener index and the
       unipolaruty value
    '''
    nSK = mol.GetNumAtoms()
    W = Wiener(mol, type='')
    unip = Unipolarity(mol)

    return 2 * W - nSK * unip


def Variation(mol, **kwargs):
    '''Obtained from the vertex distance degrees and the unipolarizability
       values
    '''
    dm = rdmolops.GetDistanceMatrix(mol)
    atom_distance_degrees = dm.sum(axis = 1)
    var_list = []

    for degree in atom_distance_degrees:
        var_list.append(degree - Unipolarity(mol))

    return max(var_list)


def CentricIndex(mol, **kwargs):
    '''The Balaban centric index (BAC) is derived for a H-depleted molecular
       graph based on the pruning of the graph, a stepwise procedurefor removing
       all the terminal vertices, i.e. vertices with a vertex degree of one, and
       the corresponding incident edges. The vertices removed at the kth step
       are nk and the Balaban centric index is calculated as the sum of the
       squares of nk numbers over the total number of steps to remove all vertices

       IMPORTANT NOTE: Dragon implementation is also implemented for cyclic
       graphs, where they consider the cycle to count as a terminal atom (+1).
       In here, the cycle is not taken into account (therefore yielding the
       value of Dragon - 1).
    '''
    desc_fn_dict = {
        'BAC': lambda nk: nk ** 2,
        'Lop': lambda nk: -(nk/nSK) * log(nk/nSK, 2)
    }
    desc_fn = desc_fn_dict[kwargs['type']]

    # Needed to exit the while loop (avoid isomers)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))

    result = 0
    vertex_degrees = \
        {atom.GetIdx(): atom.GetDegree() for atom in mol.GetAtoms()}

    # WARNING: this number is the original number of non-hydrogen atoms
    nSK = mol.GetNumAtoms()

    while 1 in vertex_degrees.values():
        nk = list(vertex_degrees.values()).count(1)

        result += desc_fn(nk)

        # Create a copy with only single bonds (to avoid sanitizing errors)
        single_bonds_mol = Chem.Mol(mol)

        # Change bond type in order to avoid errors while sanitizing
        for bond in single_bonds_mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)

        emol = Chem.EditableMol(single_bonds_mol)

        # Remove terminal atoms (degree == 1) by replacing by hydrogens and
        # eliminating them
        for atom_idx, degree in vertex_degrees.items():
            if degree == 1:
                emol.ReplaceAtom(atom_idx, Chem.Atom(1))

                query_atom = mol.GetAtomWithIdx(atom_idx)

                mol = emol.GetMol()
                # print(Chem.MolToSmiles(mol))
                Chem.SanitizeMol(mol)

        # WARNING: avoid [H][H] errors (function does not eliminate them,
        # [H]=[H] and so on...)
        atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        if sum(atomic_nums) == len(atomic_nums):
            break

        # print(Chem.MolToSmiles(mol))
        mol = Chem.RemoveHs(mol)

        # Update degrees list after eliminating terminal atoms
        vertex_degrees = \
            {atom.GetIdx(): atom.GetDegree() for atom in mol.GetAtoms()}

        # If only one atoms is left, its vertex degree is going to be 0
        if len(vertex_degrees.keys()) == 1:
            result += desc_fn(1)
            break

    return result


def TIE(mol, **kwargs):
    # WARNING: depends on intrinsic_state

    nBO = len(bonds)
    nCIC = rdMolDescriptors.CalcNumRings(mol)

    result = 0
    for bond in mol.GetBonds():
        begin_val = ValenceVertexDegree(bond.GetBeginAtom())
        end_val = ValenceVertexDegree(bond.GetEndAtom())

    return (nBO / (nCIC + 1)) * np.sqrt(result)


def Topological_all(mol):


    desc_df = pd.DataFrame()

    valence_types = ['', 'V']
    avg_types = ['', 'A']



    for order in list(range(1,3)):
        for type in valence_types:
            kwargs = {'order': order, 'type': type}
            desc_name = 'ZM{}{}'.format(order, type)
            desc_value = ZM(mol, **kwargs)

            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    Nar_types = ['S', 'H', 'G']

    for type in Nar_types:
        kwargs = {'type': type}
        desc_name = '{}Nar'.format(type)
        desc_value = Nar(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    DistanceMatrix_types = ['LPRS', 'VDA', 'MDDD']

    for type in DistanceMatrix_types:
        kwargs = {'type': type}
        desc_name = '{}'.format(type)
        desc_value = DistanceMatrix_row(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # SMTI descriptrs
    for type in valence_types:
        kwargs = {'type': type}
        desc_name = 'SMTI{}'.format(type)
        desc_value = SMTI(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # GMTI Descriptors
    for type in valence_types:
        kwargs = {'type': type}
        desc_name = 'GMTI{}'.format(type)
        desc_value = GMTI(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    # Wiener descriptor
    for type in avg_types:
        kwargs = {'type': type}
        desc_name = 'W{}'.format(type)
        desc_value = Wiener(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Whet and Jhet descriptors
    weight_props = ['Z', 'm', 'v', 'e', 'p']

    for weight in weight_props:
        kwargs = {'weight': weight}
        desc_name = 'Whet{}'.format(weight)
        desc_value = Whetw(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    for weight in weight_props:
        kwargs = {'weight': weight}
        desc_name = 'Jhet{}'.format(weight)
        desc_value = Jhetw(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Har Descriptors
    for order in list(range(1,3)):
        kwargs = {'order': order}
        desc_name = 'Har{}'.format(order)
        desc_value = Har(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Mohar descriptors TI1 and TI2
    for order in list(range(1,3)):
        kwargs = {'order': order}
        desc_name = 'TI{}'.format(order)
        desc_value = MoharIndex(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # HyDp Descriptors

    reciprocal_types = ['', 'R']

    for type in reciprocal_types:
        kwargs = {'type': type}
        desc_name = '{}HyDp'.format(type)
        desc_value = HyDp(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Hyper detour descriptors ww and Rww
    for type in reciprocal_types:
        kwargs = {'type': type}
        desc_name = '{}ww'.format(type)
        desc_value = HyperDetour(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Distance detour index descriptors
    for length in list(range(3, 13)):
        kwargs = {'length': length}
        desc_name = 'D/Dr{}'.format(length)
        desc_value = DistanceDetourIndex(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Kier shape index (Skk) Descriptors
    for order in list(range(0, 4)):
        kwargs = {'order': order}
        desc_name = 'S{}K'.format(order)
        desc_value = SkK(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    # PWk descriptors
    for order in list(range(2,6)):
        kwargs = {'order': order}
        desc_name = 'PW{}'.format(order)
        desc_value = PWk(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # ECC descriptors
    for type in ['E', 'AE', 'DE']:
        kwargs = {'type': type}
        desc_name = '{}CC'.format(type)
        desc_value = ECC(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # TODO: Can't kekulize mols with pyrrole or carbazole
    # Centric index descriptors BAC and Lop
    # for type in ['BAC', 'Lop']:
    #     kwargs = {'type': type}
    #     desc_name = '{}'.format(type)
    #     desc_value = CentricIndex(mol, **kwargs)
    #
    #     desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})
    #
    #     desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    # TODO: Check Dz (implmentation), PWk (values wrong) and TIE (not implmented yet)

    # Calculate all without changing kwargs

    other_desc = {
        'Qindex': Qindex(mol, **{}),
        'Xt': Xt(mol, **{}),
        'Ram': Ram(mol, **{}),
        'Pol': Pol(mol, **{}),
        'MSD': MSD(mol, **{}),
        'Xu': Xu(mol, **{}),
        'SPI': SPI(mol, **{}),
        'J': J(mol, **{}),
        'QW': QW(mol, **{}),
        'STN': STN(mol, **{}),
        'DetourIndex': DetourIndex(mol, **{}),
        'D/D': DistanceDetourIndex(mol, **{'length':''}),
        'Wap': Wap(mol, **{}),
        'Phi': Phi(mol, **{}),
        'Bli': Bli(mol, **{}),
        'PJI2': PJI2(mol, **{}),
        'MSD': MSD(mol, **{}),
        'DELS': DELS(mol, **{}),
        'MAXDN': MAXDN(mol, **{}),
        'MAXDP': MAXDP(mol, **{}),
        'Unipolarity': Unipolarity(mol, **{}),
        'Centralization': Centralization(mol, **{}),
        'Variation': Variation(mol, **{})

    }

    for desc_name, desc_value in other_desc.items():
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})
        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    return desc_df
