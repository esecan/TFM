import numpy as np
import pandas as pd

import subprocess

from collections import Counter
from itertools import chain

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP

from mordred import Calculator, Polarizability
# from mordred.CPSA import TPSA
from mordred.TopoPSA import TopoPSA

from utils.json_files import load_json

ATOM_CENTRED_SMARTS = load_json('descriptors/config/atom_centred.json')
ATOM_CENTRED_VALUES = pd.read_csv('descriptors/config/atom_centred.csv', header=0, encoding='UTF-8')


def Ui(mol, **kwargs):
    '''Unsaturation index taking into account double, triple and aromatic bonds
    '''
    bond_types = [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]
    bond_counter = Counter(bond_types)

    multiple_bonds = sum(
        [value for key, value in bond_counter.items() if key in [2, 3, 1.5]]
    )

    return np.log2(1 + multiple_bonds)


def Hy(mol, **kwargs):
    '''Hydrophilic factor

    NOTE: in Dragon, NH2 is counted as two Hydrophilic groups while in this
    implementation it is counted as a single one.
    '''
    nSK = len(mol.GetAtoms())
    nC  = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]')))

    mol = Chem.AddHs(mol)

    smarts = ['[#1][#8]', '[#1][#7]', '[#1][#16]']
    smarts = [Chem.MolFromSmarts(sm) for sm in smarts]

    nHy = sum([len(list(chain(mol.GetSubstructMatches(sm)))) for sm in smarts])

    return ((1 + nHy)*np.log2(1 + nHy) + nC*((1/nSK)*np.log2(1/nSK)) + \
            np.sqrt(nHy/(nSK**2))) / np.log2(1 + nSK)





def TPSA(mol, **kwargs):
    calc = Calculator(TopoPSA(no_only=kwargs['no_only']))

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]

def CrippenLogP(mol, **kwargs):
    # Wildman-Crippen LogP value (rdkit)
    if kwargs['order'] == 1:
        return MolLogP(mol)

    elif kwargs['order'] == 2:
        return MolLogP(mol) ** 2

    else:
        raise NotImplementedError

def CrippenMR(mol, **kwargs):
    # Wildman-Crippen LogP value (rdkit)
    return MolLogP(mol)

def BLTF96(mol, **kwargs):
    # Based on the Moriguchi LogP but we are using the crippen rdkit implmentation

    return -0.85 * CrippenLogP(mol, order = 1) - 1.39


def BLTD48(mol, **kwargs):
    # Based on the Moriguchi LogP but we are using the crippen rdkit implmentation

    return -0.95 * CrippenLogP(mol, order = 1) - 1.32


def ALOGP_cdk(mol, **kwargs):
    '''Crippen octanol-water partition coeff. (logP)
    '''
    smiles = Chem.MolToSmiles(mol)
    cmd = 'jython -Dpython.path=descriptors/dependencies/cdk/cdk-2.1.1.jar descriptors/cdk.py {}'.format(smiles)
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, shell=True)

    # result is a string of a list
    desc = result.stdout.decode('utf-8').split(',')

    desc = desc[0]

    if kwargs['squared']:
        desc = desc**2

    return desc


def MLOGP_old(mol, **kwargs):
    '''Moriguchi model based on structural parameters)
    '''

    # FCX = Summation of number of carbon and halogen atoms weighted
    FCX = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [6,17]:
            FCX += 1.0
        elif atom.GetAtomicNum() == 9:
            FCX += 0.5
        elif atom.GetAtomicNum() == 35:
            FCX += 1.5
        elif atom.GetAtomicNum() == 53:
            FCX += 2.0

    # print('FCX', FCX)
    # NO + NN = Total number of nitrogen and oxygen atoms
    # print(NONN)
    # FPRX = Proximity effect of N/O: X-Y=2, X-A-Y=1 (X, Y: N/O, A: C, S, or P) with
    #       correction -1 for carbox-amide/sulfonamide
    # NUNS = Total number of unsaturated bonds (not those in NO2)

    NUNS = 0
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            NUNS += 1

    smarts = Chem.MolFromSmarts('O[N]=O')
    NUNS -= len(mol.GetSubstructMatches(smarts))

    # print(NUNS)
    # IHB = Dummy variable for the presence of intramolecular H-bonds

    return NUNS


def MLOGP(mol, **kwargs):
    '''Moriguchi MLogP
    '''
    smiles = Chem.MolToSmiles(mol)
    cmd = 'jython -Dpython.path=descriptors/dependencies/MLogP.jar descriptors/cdk.py {}'.format(smiles)
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, shell=True)

    # result is a string of a list
    desc = result.stdout.decode('utf-8').split(',')

    if kwargs['squared']:
        desc = desc**2

    return desc


def Lipinski(mol, **kwargs):
    '''Implementation of the Lipinski Alert Index (rule of 5), that predicts
    that poor absorption or permeation is more likely when:
        - there are more than 5 H-bond donors
        - there are more than 10 H-bond acceptors
        - the molecular weight is over 500
        - the Moriguchi's logP is over 4.15
    The 'H-bond donors' property is determined by adding all of the hydrogens
    bonded to Os and Ns. The 'H-bond acceptors' property is determined by
    adding all of the Os and Ns in the molecule.

    This computational alert is a filter that identifies compounds lying in a
    region of property space where the probability of useful oral activity is
    very low. A compound that fails the alert will likely be poorly
    bioavailable because of poor absorption or permeation.

    The alert index is a dummy variable taking value 1 when two or more
    properties are out of range.
    '''
    donors = Chem.MolFromSmarts('[#8,#7][#1]')
    acceptors = Chem.MolFromSmarts('[#8,#7]')

    h_bond_donors = len(list(chain(mol.GetSubstructMatches(donors))))
    h_bond_acceptors = len(list(chain(mol.GetSubstructMatches(acceptors))))
    mw = ExactMolWt(mol)
    mlogp = MLOGP(mol)

    rule = 0
    if h_bond_donors > 5: rule += 1
    if h_bond_acceptors > 10: rule += 1
    if mw > 500: rule += 1
    if mlogp > 4.15: rule += 1

    return int(rule >= 2)


def atom_centred_based(mol, **kwargs):
    col_idx = ATOM_CENTRED_VALUES.columns.get_loc(kwargs['type'])

    result = 0.0
    for symbol, smarts in ATOM_CENTRED_SMARTS.items():
        for sm in smarts:
            # print(Chem.MolToSmiles(mol), symbol, sm, mol.HasSubstructMatch(Chem.MolFromSmarts(sm)))
            if not mol.HasSubstructMatch(Chem.MolFromSmarts(sm)):
                continue

            # Filter by SMARTS
            values = ATOM_CENTRED_VALUES[ATOM_CENTRED_VALUES['Symbol'] == symbol]

            # Filter by type: AMR (MR) or ALOGP (hydrophobicity)
            try:
                result += float(values.iloc[0, col_idx])
            except ValueError:
                continue

    return result


def GVWAI_old(mol, **kwargs):
    '''Drug like index (dummy variable) cosidering 4 descriptors, alogp, MW, AMR
    and number of atoms, if all criteria are sastisfied, 1, otherwise, 0.
    '''
    # AMR between 40 and 130
    # Get the molecule polarizability

    calc = Calculator(
        Polarizability.APol(True)
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    polarizability = desc.iloc[0, 0]


    AMR = (4/3)*pi*polarizability

    if AMR < 40 or AMR > 130:
        return 0

    # ALOGP between -0.4 and 4.6
    ALOGP()

    # elif ALOGP(mol) < -0.4 or ALOGP(mol) > 5.6:
    #     print("AAA")
    #     return 0

    # Number of atoms between 20 and 70 (considering Hs)
    if mol.GetNumAtoms() < 20 or mol.GetNumAtoms() > 70:
        return 0

    # MW between 160 and 480
    elif Chem.Descriptors.ExactMolWt(mol) < 160 or Chem.Descriptors.ExactMolWt(mol) > 480:
        return 0

    else:
        return 1


def GVWAI(mol, **kwargs):
    '''Drug like index (dummy variable) cosidering 4 descriptors, alogp, MW, AMR
    and number of atoms, if all criteria are sastisfied, 1, otherwise, 0.

    Keyword arguments:
        index (string) -- GVWAI, Inflammat, Depressant, Psychotic, Hypertens,
                          Hypnotic, Neoplastic or Infective
        percent (int)  -- Percent of chemical space to take into account
    '''
    drug_like_indices = {
        'GVWAI-80': {'ALOGP': [-0.4, 5.6], 'AMR': [40, 130], 'MW': [160, 480], 'Atoms': [20, 70]},
        'GVWAI-50': {'ALOGP': [1.3, 4.1], 'AMR': [70, 110], 'MW': [230, 390], 'Atoms': [30, 55]},
        'Inflammat-80': {'ALOGP': [1.4, 4.5], 'AMR': [59, 119], 'MW': [212, 447], 'Atoms': [24, 59]},
        'Inflammat-50': {'ALOGP': [2.6, 4.2], 'AMR': [67, 97], 'MW': [260, 380], 'Atoms': [28, 40]},
        'Depressant-80': {'ALOGP': [1.4, 4.9], 'AMR': [62, 114], 'MW': [210, 380], 'Atoms': [32, 56]},
        'Depressant-50': {'ALOGP': [2.1, 4.0], 'AMR': [75, 95], 'MW': [260, 330], 'Atoms': [37, 48]},
        'Psychotic-80': {'ALOGP': [2.3, 5.2], 'AMR': [85, 131], 'MW': [274, 464], 'Atoms': [40, 63]},
        'Psychotic-50': {'ALOGP': [3.3, 5.0], 'AMR': [94, 120], 'MW': [322, 422], 'Atoms': [49, 61]},
        'Hypertens-80': {'ALOGP': [-0.5, 4.5], 'AMR': [54, 128], 'MW': [206, 506], 'Atoms': [28, 66]},
        'Hypertens-50': {'ALOGP': [1, 3.4], 'AMR': [68, 116], 'MW': [281, 433], 'Atoms': [36, 58]},
        'Hypnotic-80': {'ALOGP': [0.5, 3.9], 'AMR': [43, 97], 'MW': [162, 360], 'Atoms': [20, 45]},
        'Hypnotic-50': {'ALOGP': [1.3, 3.5], 'AMR': [43, 73], 'MW': [212, 306], 'Atoms': [29, 38]},
        'Neoplastic-80': {'ALOGP': [-1.5, 4.7], 'AMR': [43, 128], 'MW': [180, 475], 'Atoms': [21, 63]},
        'Neoplastic-50': {'ALOGP': [0.0, 3.7], 'AMR': [60, 107], 'MW': [258, 388], 'Atoms': [30, 55]},
        'Infective-80': {'ALOGP': [-0.3, 5.1], 'AMR': [44, 144], 'MW': [145, 455], 'Atoms': [12, 64]},
        'Infective-50': {'ALOGP': [0.8, 3.8], 'AMR': [68, 138], 'MW': [192, 392], 'Atoms': [12, 42]},
    }

    desc_name = kwargs['index'] + '-' + kwargs['percent']
    index_dict = drug_like_indices[desc_name]

    mol = Chem.RemoveHs(mol)

    mol_dict = {
        'ALOGP': ALOGP(mol, {}),
        'AMR': AMR(mol, {}),
        'MW': Descriptors.ExactMolWt(mol),
        'Atoms': len(mol.GetAtoms())
    }

    dummy = sum(
        [index_dict[key][0] < mol_dict[key] < index_dict[key][1]]
        for key in mol_dict.keys()
    )

    return int(dummy == len(mol_dict.keys()))
