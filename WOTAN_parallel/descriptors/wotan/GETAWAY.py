

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, SaltRemover

import pandas as pd

GETAWAYNAMES=['ITH','ISH','HIC','HGM','H0u','H1u','H2u','H3u','H4u','H5u','H6u','H7u','H8u','HTu',
'HATS0u','HATS1u','HATS2u','HATS3u','HATS4u','HATS5u','HATS6u','HATS7u','HATS8u','HATSu','H0m','H1m','H2m','H3m','H4m','H5m',
'H6m','H7m','H8m','HTm','HATS0m','HATS1m','HATS2m','HATS3m','HATS4m','HATS5m','HATS6m','HATS7m','HATS8m','HATSm','H0v','H1v',
'H2v','H3v','H4v','H5v','H6v','H7v','H8v','HTv','HATS0v','HATS1v','HATS2v','HATS3v','HATS4v','HATS5v','HATS6v','HATS7v','HATS8v',
'HATSv','H0e','H1e','H2e','H3e','H4e','H5e','H6e','H7e','H8e','HTe','HATS0e','HATS1e','HATS2e','HATS3e','HATS4e','HATS5e','HATS6e',
'HATS7e','HATS8e','HATSe','H0p','H1p','H2p','H3p','H4p','H5p','H6p','H7p','H8p','HTp','HATS0p','HATS1p','HATS2p','HATS3p','HATS4p',
'HATS5p','HATS6p','HATS7p','HATS8p','HATSp','H0i','H1i','H2i','H3i','H4i','H5i','H6i','H7i','H8i','HTi','HATS0i','HATS1i','HATS2i',
'HATS3i','HATS4i','HATS5i','HATS6i','HATS7i','HATS8i','HATSi','H0s','H1s','H2s','H3s','H4s','H5s','H6s','H7s','H8s','HTs','HATS0s',
'HATS1s','HATS2s','HATS3s','HATS4s','HATS5s','HATS6s','HATS7s','HATS8s','HATSs','RCON','RARS','REIG','R1u','R2u','R3u','R4u','R5u',
'R6u','R7u','R8u','RTu','R1u+','R2u+','R3u+','R4u+','R5u+','R6u+','R7u+','R8u+','RTu+','R1m','R2m','R3m','R4m','R5m','R6m','R7m',
'R8m','RTm','R1m+','R2m+','R3m+','R4m+','R5m+','R6m+','R7m+','R8m+','RTm+','R1v','R2v','R3v','R4v','R5v','R6v','R7v','R8v','RTv',
'R1v+','R2v+','R3v+','R4v+','R5v+','R6v+','R7v+','R8v+','RTv+','R1e','R2e','R3e','R4e','R5e','R6e','R7e','R8e','RTe','R1e+','R2e+',
'R3e+','R4e+','R5e+','R6e+','R7e+','R8e+','RTe+','R1p','R2p','R3p','R4p','R5p','R6p','R7p','R8p','RTp','R1p+','R2p+','R3p+','R4p+',
'R5p+','R6p+','R7p+','R8p+','RTp+','R1i','R2i','R3i','R4i','R5i','R6i','R7i','R8i','RTi','R1i+','R2i+','R3i+','R4i+','R5i+','R6i+',
'R7i+','R8i+','RTi+','R1s','R2s','R3s','R4s','R5s','R6s','R7s','R8s','RTs','R1s+','R2s+','R3s+','R4s+','R5s+','R6s+','R7s+','R8s+','RTs+']


def Getaway(mol, **kwargs):
    '''Leverage-weighted autocorrelation of lag [order] / weighted by [prop]

    Keyword arguments:
        name -- getaway descriptor name
    '''
    # suppl = Chem.SDMolSupplier('tests/hats1m/hats1m.sdf')
    # mol = suppl[0]

    seed = 429647

    remover = Chem.SaltRemover.SaltRemover(defnData="[Na,Cl]")
    mol = remover.StripMol(mol)

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    if mol.GetNumConformers() < 1:
        return 0.0

    result = rdMolDescriptors.CalcGETAWAY(mol)

    desc_name = kwargs['name']
    desc_idx = GETAWAYNAMES.index(desc_name)

    return result[desc_idx]

def GETAWAY_all(mol):
    # For this descriptors the salts must be explicitely removed
    remover = Chem.SaltRemover.SaltRemover(defnData="[Na,Cl]")
    smi = Chem.MolToSmiles(mol)
    # print(smi)
    mol = remover.StripMol(mol)

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    desc_df = pd.DataFrame()

    result = rdMolDescriptors.CalcGETAWAY(mol)

    for getaway in GETAWAYNAMES:
        desc_name = getaway
        desc_idx = GETAWAYNAMES.index(desc_name)
        desc_value = result[desc_idx]
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
