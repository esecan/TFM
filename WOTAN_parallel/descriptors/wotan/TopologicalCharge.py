
from mordred import Calculator, TopologicalCharge
import pandas as pd


def Topological_Charge(mol, **kwargs):

    '''Presence/absence of [atom] - [atom] at topological distance [distance]

    Keyword arguments:
        type -- raw, mean
        order -- length of the path (k)
    '''

    calc = Calculator(
        TopologicalCharge.TopologicalCharge(kwargs['prop'], kwargs['order'])
        )

    desc = calc.pandas([mol], nproc=1, quiet=True)
    return desc.iloc[0, 0]


def TopologicalCharge_all(mol):

    order_values = list(range(1, 11))
    properties = ['raw', 'mean']

    desc_df = pd.DataFrame()

    for prop in properties:
        for order in order_values:
            kwargs = {'prop': prop, 'order': order}
            if prop == 'raw':
                desc_name = 'GGI{}'.format(order)
            elif prop == 'mean':
                desc_name = 'JGI{}'.format(order)
            desc_value = Topological_Charge(mol, **kwargs)
            desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})

            desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    # Calculate all without order
    other_desc = {
        'JGT10': Topological_Charge(mol, **{'prop': 'global', 'order': 10})
    }

    for desc_name, desc_value in other_desc.items():
        desc_value = pd.DataFrame.from_dict({desc_name: [desc_value]})
        desc_df = pd.concat([desc_df, desc_value], axis=1, sort=False)

    return desc_df
