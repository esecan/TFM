from mordred import Calculator
from mordred import InformationContent

import pandas as pd

def not_implemented_desc(mol, **kwargs):

    return 0


def ICk(mol, **kwargs):
    '''The Information Content (ICk) measures the deviation of
    the information content ICk from its maximum value
    '''
    calc = Calculator(
    InformationContent.InformationContent(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]

def tICk(mol, **kwargs):

    calc = Calculator(
    InformationContent.TotalIC(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]


def SICk(mol, **kwargs):

    calc = Calculator(
        InformationContent.StructuralIC(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]

def BICk(mol, **kwargs):

    calc = Calculator(
        InformationContent.BondingIC(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]

def CICk(mol, **kwargs):
    '''The Complementary Information Content (CICk) measures the deviation of
    the information content ICk from its maximum value
    '''
    calc = Calculator(
        InformationContent.ComplementaryIC(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]

def MICk(mol, **kwargs):

    calc = Calculator(
        InformationContent.ModifiedIC(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]

def ZMICk(mol, **kwargs):

    calc = Calculator(
        InformationContent.ZModifiedIC(kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    return desc.iloc[0, 0]


def InformationIndices_all(mol):

    information_descriptors = {
        'IC': ICk,
        'tIC': tICk,
        'SIC': SICk,
        'BIC': BICk,
        'CIC': CICk,
        'MIC': MICk,
        'ZMIC': ZMICk
    }

    order_values = [0, 1, 2, 3, 4, 5]

    desc_df = pd.DataFrame()

    for desc in information_descriptors.keys():
        for order in order_values:
            kwargs = {'order': order}
            try:
                desc_name = '{}{}'.format(desc, order)
                desc_value = information_descriptors[desc](mol, **kwargs)
                desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

                desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

            except: continue

    return desc_df
