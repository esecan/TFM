
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import pandas as pd

RDFNAMES = ['RDF010u', 'RDF015u', 'RDF020u', 'RDF025u', 'RDF030u', 'RDF035u', 'RDF040u', 'RDF045u', 'RDF050u',
'RDF055u', 'RDF060u', 'RDF065u', 'RDF070u', 'RDF075u', 'RDF080u', 'RDF085u', 'RDF090u', 'RDF095u',
'RDF100u', 'RDF105u', 'RDF110u', 'RDF115u', 'RDF120u', 'RDF125u', 'RDF130u', 'RDF135u', 'RDF140u',
'RDF145u', 'RDF150u', 'RDF155u',
'RDF010m', 'RDF015m', 'RDF020m', 'RDF025m', 'RDF030m', 'RDF035m', 'RDF040m', 'RDF045m', 'RDF050m',
'RDF055m', 'RDF060m', 'RDF065m', 'RDF070m', 'RDF075m', 'RDF080m', 'RDF085m', 'RDF090m', 'RDF095m',
'RDF100m', 'RDF105m', 'RDF110m', 'RDF115m', 'RDF120m', 'RDF125m', 'RDF130m', 'RDF135m', 'RDF140m',
'RDF145m', 'RDF150m', 'RDF155m',
'RDF010v', 'RDF015v', 'RDF020v', 'RDF025v', 'RDF030v', 'RDF035v', 'RDF040v', 'RDF045v', 'RDF050v',
'RDF055v', 'RDF060v', 'RDF065v', 'RDF070v', 'RDF075v', 'RDF080v', 'RDF085v', 'RDF090v', 'RDF095v',
'RDF100v', 'RDF105v', 'RDF110v', 'RDF115v', 'RDF120v', 'RDF125v', 'RDF130v', 'RDF135v', 'RDF140v',
'RDF145v', 'RDF150v', 'RDF155v',
'RDF010e', 'RDF015e', 'RDF020e', 'RDF025e', 'RDF030e', 'RDF035e', 'RDF040e', 'RDF045e', 'RDF050e',
'RDF055e', 'RDF060e', 'RDF065e', 'RDF070e', 'RDF075e', 'RDF080e', 'RDF085e', 'RDF090e', 'RDF095e',
'RDF100e', 'RDF105e', 'RDF110e', 'RDF115e', 'RDF120e', 'RDF125e', 'RDF130e', 'RDF135e', 'RDF140e',
'RDF145e', 'RDF150e', 'RDF155e',
'RDF010p', 'RDF015p', 'RDF020p', 'RDF025p', 'RDF030p', 'RDF035p', 'RDF040p', 'RDF045p', 'RDF050p',
'RDF055p', 'RDF060p', 'RDF065p', 'RDF070p', 'RDF075p', 'RDF080p', 'RDF085p', 'RDF090p', 'RDF095p',
'RDF100p', 'RDF105p', 'RDF110p', 'RDF115p', 'RDF120p', 'RDF125p', 'RDF130p', 'RDF135p', 'RDF140p',
'RDF145p', 'RDF150p', 'RDF155p',
'RDF010i', 'RDF015i', 'RDF020i', 'RDF025i', 'RDF030i', 'RDF035i', 'RDF040i', 'RDF045i', 'RDF050i',
'RDF055i', 'RDF060i', 'RDF065i', 'RDF070i', 'RDF075i', 'RDF080i', 'RDF085i', 'RDF090i', 'RDF095i',
'RDF100i', 'RDF105i', 'RDF110i', 'RDF115i', 'RDF120i', 'RDF125i', 'RDF130i', 'RDF135i', 'RDF140i',
'RDF145i', 'RDF150i', 'RDF155i',
'RDF010s', 'RDF015s', 'RDF020s', 'RDF025s', 'RDF030s', 'RDF035s', 'RDF040s', 'RDF045s', 'RDF050s',
'RDF055s', 'RDF060s', 'RDF065s', 'RDF070s', 'RDF075s', 'RDF080s', 'RDF085s', 'RDF090s', 'RDF095s',
'RDF100s', 'RDF105s', 'RDF110s', 'RDF115s', 'RDF120s', 'RDF125s', 'RDF130s', 'RDF135s', 'RDF140s',
'RDF145s', 'RDF150s', 'RDF155s']

WHIMNAMES = ['L1u', 'L1m', 'L1v', 'L1e', 'L1p', 'L1i', 'L1s', 'L2u', 'L2m', 'L2v', 'L2e', 'L2p',
'L2i', 'L2s', 'L3u', 'L3m', 'L3v', 'L3e', 'L3p', 'L3i', 'L3s', 'P1u', 'P1m', 'P1v',
'P1e', 'P1p', 'P1i', 'P1s', 'P2u', 'P2m', 'P2v', 'P2e', 'P2p', 'P2i', 'P2s', 'G1u',
'G1m', 'G1v', 'G1e', 'G1p', 'G1i', 'G1s', 'G2u', 'G2m', 'G2v', 'G2e', 'G2p', 'G2i',
'G2s', 'G3u', 'G3m', 'G3v', 'G3e', 'G3p', 'G3i', 'G3s', 'E1u', 'E1m', 'E1v', 'E1e',
'E1p', 'E1i', 'E1s', 'E2u', 'E2m', 'E2v', 'E2e', 'E2p', 'E2i', 'E2s', 'E3u', 'E3m',
'E3v', 'E3e', 'E3p', 'E3i', 'E3s', 'Tm', 'Tv', 'Te', 'Tp', 'Ti', 'Ts', 'Tu',
'Am', 'Av', 'Ae', 'Ap', 'Ai', 'As', 'Gu', 'Gm', 'Ku', 'Km', 'Kv', 'Ke', 'Kp', 'Ki',
'Ks', 'Du', 'Dm', 'Dv', 'De', 'Dp', 'Di', 'Ds', 'Vu', 'Vm', 'Vv', 'Ve', 'Vp', 'Vi',
'Vs']

AUTOCORRNAMES = ['TDB01u', 'TDB02u', 'TDB03u', 'TDB04u', 'TDB05u', 'TDB06u', 'TDB07u', 'TDB08u',
'TDB09u', 'TDB10u', 'TDB01m', 'TDB02m', 'TDB03m', 'TDB04m', 'TDB05m', 'TDB06m',
'TDB07m', 'TDB08m', 'TDB09m', 'TDB10m', 'TDB01v', 'TDB02v', 'TDB03v', 'TDB04v',
'TDB05v', 'TDB06v', 'TDB07v', 'TDB08v', 'TDB09v', 'TDB10v', 'TDB01e', 'TDB02e',
'TDB03e', 'TDB04e', 'TDB05e', 'TDB06e', 'TDB07e', 'TDB08e', 'TDB09e', 'TDB10e',
'TDB01p', 'TDB02p', 'TDB03p', 'TDB04p', 'TDB05p', 'TDB06p', 'TDB07p', 'TDB08p',
'TDB09p', 'TDB10p', 'TDB01i', 'TDB02i', 'TDB03i', 'TDB04i', 'TDB05i', 'TDB06i',
'TDB07i', 'TDB08i', 'TDB09i', 'TDB10i', 'TDB01s', 'TDB02s', 'TDB03s', 'TDB04s',
'TDB05s', 'TDB06s', 'TDB07s', 'TDB08s', 'TDB09s', 'TDB10s', 'TDB01r', 'TDB02r',
'TDB03r', 'TDB04r', 'TDB05r', 'TDB06r', 'TDB07r', 'TDB08r', 'TDB09r', 'TDB10r']

MORSENAMES = ['Mor01u', 'Mor02u', 'Mor03u', 'Mor04u', 'Mor05u', 'Mor06u', 'Mor07u', 'Mor08u', 'Mor09u',
'Mor10u', 'Mor11u', 'Mor12u', 'Mor13u', 'Mor14u', 'Mor15u', 'Mor16u', 'Mor17u', 'Mor18u',
'Mor19u', 'Mor20u', 'Mor21u', 'Mor22u', 'Mor23u', 'Mor24u', 'Mor25u', 'Mor26u', 'Mor27u',
'Mor28u', 'Mor29u', 'Mor30u', 'Mor31u', 'Mor32u',
'Mor01m', 'Mor02m', 'Mor03m', 'Mor04m', 'Mor05m', 'Mor06m', 'Mor07m', 'Mor08m', 'Mor09m',
'Mor10m', 'Mor11m', 'Mor12m', 'Mor13m', 'Mor14m', 'Mor15m', 'Mor16m', 'Mor17m', 'Mor18m',
'Mor19m', 'Mor20m', 'Mor21m', 'Mor22m', 'Mor23m', 'Mor24m', 'Mor25m', 'Mor26m', 'Mor27m',
'Mor28m', 'Mor29m', 'Mor30m', 'Mor31m', 'Mor32m',
'Mor01v', 'Mor02v', 'Mor03v', 'Mor04v', 'Mor05v', 'Mor06v', 'Mor07v', 'Mor08v', 'Mor09v',
'Mor10v', 'Mor11v', 'Mor12v', 'Mor13v', 'Mor14v', 'Mor15v', 'Mor16v', 'Mor17v', 'Mor18v',
'Mor19v', 'Mor20v', 'Mor21v', 'Mor22v', 'Mor23v', 'Mor24v', 'Mor25v', 'Mor26v', 'Mor27v',
'Mor28v', 'Mor29v', 'Mor30v', 'Mor31v', 'Mor32v',
'Mor01e', 'Mor02e', 'Mor03e', 'Mor04e', 'Mor05e', 'Mor06e', 'Mor07e', 'Mor08e', 'Mor09e',
'Mor10e', 'Mor11e', 'Mor12e', 'Mor13e', 'Mor14e', 'Mor15e', 'Mor16e', 'Mor17e', 'Mor18e',
'Mor19e', 'Mor20e', 'Mor21e', 'Mor22e', 'Mor23e', 'Mor24e', 'Mor25e', 'Mor26e', 'Mor27e',
'Mor28e', 'Mor29e', 'Mor30e', 'Mor31e', 'Mor32e',
'Mor01p', 'Mor02p', 'Mor03p', 'Mor04p', 'Mor05p', 'Mor06p', 'Mor07p', 'Mor08p', 'Mor09p',
'Mor10p', 'Mor11p', 'Mor12p', 'Mor13p', 'Mor14p', 'Mor15p', 'Mor16p', 'Mor17p', 'Mor18p',
'Mor19p', 'Mor20p', 'Mor21p', 'Mor22p', 'Mor23p', 'Mor24p', 'Mor25p', 'Mor26p', 'Mor27p',
'Mor28p', 'Mor29p', 'Mor30p', 'Mor31p', 'Mor32p',
'Mor01i', 'Mor02i', 'Mor03i', 'Mor04i', 'Mor05i', 'Mor06i', 'Mor07i', 'Mor08i', 'Mor09i',
'Mor10i', 'Mor11i', 'Mor12i', 'Mor13i', 'Mor14i', 'Mor15i', 'Mor16i', 'Mor17i', 'Mor18i',
'Mor19i', 'Mor20i', 'Mor21i', 'Mor22i', 'Mor23i', 'Mor24i', 'Mor25i', 'Mor26i', 'Mor27i',
'Mor28i', 'Mor29i', 'Mor30i', 'Mor31i', 'Mor32i',
'Mor01s', 'Mor02s', 'Mor03s', 'Mor04s', 'Mor05s', 'Mor06s', 'Mor07s', 'Mor08s', 'Mor09s',
'Mor10s', 'Mor11s', 'Mor12s', 'Mor13s', 'Mor14s', 'Mor15s', 'Mor16s', 'Mor17s', 'Mor18s',
'Mor19s', 'Mor20s', 'Mor21s', 'Mor22s', 'Mor23s', 'Mor24s', 'Mor25s', 'Mor26s', 'Mor27s',
'Mor28s', 'Mor29s', 'Mor30s', 'Mor31s', 'Mor32s']


def PBF(mol, **kwargs):
    '''Returns the PBF (plane of best fit) descriptor
    '''
    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcPBF(mol)

    return result

def PMI(mol, **kwargs):
    '''Returns the principal moments of inertia
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    desc_name = 'CalcPMI{}'.format(kwargs['order'])

    desc_fn = getattr(rdMolDescriptors, desc_name)
    result = desc_fn(mol)

    return result

def RDF(mol, **kwargs):
    '''Returns the PBF (plane of best fit) descriptor
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcRDF(mol)

    desc_name = kwargs['order']
    desc_idx = RDFNAMES.index(desc_name)

    return result[desc_idx]

def NPR(mol, **kwargs):
    '''Returns the Normalized principal moments ratios
    '''

    NPR_orders = [1, 2]

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    desc_name = 'CalcNPR{}'.format(kwargs['order'])

    desc_fn = getattr(rdMolDescriptors, desc_name)
    result = desc_fn(mol)

    return result


def RadiusGyration(mol, **kwargs):
    '''Returns the Radius of gyration of the molecule
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcRadiusOfGyration(mol)

    return result

def Spherocity(mol, **kwargs):
    '''Returns the Radius of gyration of the molecule
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcSpherocityIndex(mol)

    return result


def WHIM(mol, **kwargs):

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    if mol.GetNumConformers() < 1:
        return 0.0

    result = rdMolDescriptors.CalcWHIM(mol)

    desc_name = kwargs['order']
    desc_idx = WHIMNAMES.index(desc_name)

    return result[desc_idx]


def InertialShapeFactor(mol, **kwargs):
    '''Returns the Inertial Shape factor of the molecule
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcInertialShapeFactor(mol)

    return result

def Eccentricity(mol, **kwargs):
    '''Returns the Inertial Shape factor of the molecule
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcEccentricity(mol)

    return result

def Asphericity(mol, **kwargs):
    '''Returns the Inertial Shape factor of the molecule
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcAsphericity(mol)

    return result

def Autocorr3D(mol, **kwargs):
    '''Returns the Radius of gyration of the molecule
    '''

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    result = rdMolDescriptors.CalcAUTOCORR3D(mol)

    desc_name = kwargs['order']
    desc_idx = AUTOCORRNAMES.index(desc_name)

    return result[desc_idx]


def Morse(mol, **kwargs):

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    if mol.GetNumConformers() < 1:
        return 0.0

    result = rdMolDescriptors.CalcMORSE(mol)

    desc_name = kwargs['order']
    desc_idx = MORSENAMES.index(desc_name)

    return result[desc_idx]


def Rdkit3D_all(mol):

    desc_df = pd.DataFrame()

    PMI_orders = [1, 2, 3]
    NPR_orders = [1, 2]

    seed = 429647

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)

    for order in PMI_orders:

        kwargs = {'order': order}
        desc_name = 'PMI{}'.format(order)
        desc_value = PMI(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    for order in NPR_orders:

        kwargs = {'order': order}
        desc_name = 'NPR{}'.format(order)
        desc_value = NPR(mol, **kwargs)

        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)



    result = rdMolDescriptors.CalcRDF(mol)

    for rdf in RDFNAMES:
        desc_name = rdf
        desc_idx = RDFNAMES.index(desc_name)
        desc_value = result[desc_idx]
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    result = rdMolDescriptors.CalcWHIM(mol)

    for whim in WHIMNAMES:
        desc_name = whim
        desc_idx = WHIMNAMES.index(desc_name)
        desc_value = result[desc_idx]
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    result = rdMolDescriptors.CalcAUTOCORR3D(mol)

    for autocorr in AUTOCORRNAMES:
        desc_name = autocorr
        desc_idx = AUTOCORRNAMES.index(desc_name)
        desc_value = result[desc_idx]
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    result = rdMolDescriptors.CalcMORSE(mol)

    for morse in MORSENAMES:
        desc_name = morse
        desc_idx = MORSENAMES.index(desc_name)
        desc_value = result[desc_idx]
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    # Calculate all without changing kwargs
    other_desc = {
        'PBF': PBF(mol, **{}),
        'RadiusGyration': RadiusGyration(mol, **{}),
        'Spherocity': Spherocity(mol, **{}),
        'InertialShapeFactor': InertialShapeFactor(mol, **{}),
        'Eccentricity': Eccentricity(mol, **{}),
        'Asphericity': Asphericity(mol, **{})

    }

    for desc_name, desc_value in other_desc.items():
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})
        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
