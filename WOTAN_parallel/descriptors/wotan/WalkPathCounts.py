import numpy as np
import pandas as pd

from mordred import Calculator, WalkCount, PathCount

from math import log, exp


def WCk(mol, **kwargs):
    '''Self-returning walk count of order [order]

    Keyword arguments:
        order -- order of the walk count
        total -- whether it is total path count or not
                 - false for MWC (order 1-10)
                 - false for SRW (order 2-10)
                 - true fopr TMWC10 and TSRW10

        return -- whether it is self-returning path count or not (False for MWC)
                  - false for MWC and TMWC10
                  - true for SRW and TSRW10
    '''
    calc = Calculator(
        WalkCount.WalkCount(
            kwargs['order'], kwargs['total'], kwargs['self_returning']
        )
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    try:
        # Mordred calculates log(An.trace() + 1), being An the AdjacencyMatrix
        # Therefore, exponentiation and substraction of 1 must be performed
        if kwargs['self_returning']:
            return int(np.expm1(desc.iloc[0, 0]))

        else:
            # Use float to avoid errors from Mordred
            return float(desc.iloc[0, 0])

    # One-atom molecules
    except:
        return 0


def TWC(mol, **kwargs):
    calc = Calculator(
        WalkCount.WalkCount(order=10, total=True, self_returning=False)
    )

    return calc.pandas([mol], nproc=1, quiet=True).iloc[0, 0]


def PCk(mol, **kwargs):
    '''Molecular (multiple) path count of order kwargs[order]

        - piPCk (order 1-10) with Pi=true, total=false, log=True
        - TpiPC Pi=true, total=true, log=True
        - MPC (order 2-10) with Pi=false, total=false, log=false
        - TpiPC Pi=true, total=true, log=True
        - TMPC10 with Pi=false, total=true, log=false
    '''
    calc = Calculator(
        PathCount.PathCount(
            kwargs['order'], kwargs['pi'], kwargs['total'], kwargs['log']
        )
    )

    return calc.pandas([mol], nproc=1, quiet=True).iloc[0,0]


def PCR_PCD(mol, **kwargs):
    '''Function used to compute TPC, piID, PCR, PCD
    '''
    # Assume maximum order is the length of the molecule. Otherwise, check RDKit
    # for calculating the maximum path

    # TODO: split into two different functions
    # TODO: avoid lambda transf by using the log=True argument

    tpc = TPC(mol, **{})
    piid = piID(mol, **{})

    if kwargs['type'] == 'PCR':
        return piid / tpc

    elif kwargs['type'] == 'PCD':
        return piid - tpc

    else:
        raise NotImplementedError


def TPC(mol, **kwargs):
    calc = Calculator(
        PathCount.PathCount(order=10, pi=False, total=True, log=True)
    )

    return calc.pandas([mol], nproc=1, quiet=True).iloc[0, 0]


def piID(mol, **kwargs):
    calc = Calculator(
        PathCount.PathCount(order=10, pi=True, total=True, log=True)
    )

    return calc.pandas([mol], nproc=1, quiet=True).iloc[0, 0]


def WalkPathCounts_all(mol):

    desc_df = pd.DataFrame()

    walk_dict = {'MWC': False, 'SRW': True} # value indicates 'self_returning'
    path_dict = {'MPC': False, 'piPC': True} # value indicates 'pi'

    for order in range(1, 11):
        for key, value in walk_dict.items():
            kwargs = {
                'order': order,
                'total': False,
                'self_returning': value
            }

            desc_name = '{}{:02d}'.format(key, order)

            desc_value = WCk(mol, **kwargs)
            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

        for key, value in path_dict.items():
            kwargs = {
                'order': order,
                'pi': value,
                'total': False,
                'log': value,
            }

            desc_name = '{}{:02d}'.format(key, order)

            desc_value = PCk(mol, **kwargs)
            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    # Calculate all without order
    other_desc = {
        'TWC': TWC(mol, **{}),
        'TPC': TPC(mol, **{}),
        'piID': piID(mol, **{}),
        'PCR': PCR_PCD(mol, **{'type': 'PCR'}),
        'PCD': PCR_PCD(mol, **{'type': 'PCD'}),
    }

    for desc_name, desc_value in other_desc.items():
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})
        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)


    return desc_df
