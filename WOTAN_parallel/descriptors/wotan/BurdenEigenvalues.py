
from mordred import Calculator, BCUT

import numpy as np
import pandas as pd

def BELpk(mol, **kwargs):
    '''eigenvalue n. k of Burden matrix weighted by property
                    (c, dv d, s, Z, m, v se, pe, are, p i, )
    '''

## Mordred direct timplementation

    calc = Calculator(
        BCUT.BCUT(kwargs['prop'], kwargs['order'])
    )

    desc = calc.pandas([mol], nproc=1, quiet=True)

    if type(desc.iloc[0, 0]) is np.float64:
        return(desc.iloc[0, 0])

    else: return 0

def BurdenEigenvalues_all(mol):

    order_values = [0, -1]
    properties = ['c', 'dv', 'd', 's', 'Z', 'm', 'v', 'se', 'pe', 'are', 'p', 'i']

    desc_df = pd.DataFrame()

    for prop in properties:
        for order in order_values:
            kwargs = {'prop': prop, 'order': order}

            desc_name = 'BEL{}{}'.format(prop, order)
            desc_value = BELpk(mol, **kwargs)
            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
