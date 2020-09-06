# -*- coding: utf-8 -*-

import re
import time

from requests import get
import pandas as pd


def cas_to_smiles(cas_list, sleep=0.2, limit=None):
    '''Retrieves SMILES from CAS using the NCBI's CACTUS web server.

    Keyword arguments:
        cas_list (list) -- single entry/list of CAS numbers to convert to SMILES
        sleep (float) -- seconds to sleep between attempts
        limit (int) -- limits the amount of CAS converted to avoid IP blocking

    Output:
        None -- Writes converted CAS to file
    '''
    error = 'Page not found (404)'

    n = 1
    smiles_list = []

    # Avoid error while iterating a single element
    if not type(cas_list) is list:
        cas_list = [cas_list]

    # Force the iteration in order to control IP blocking
    for cas in cas_list:
        # Sleep to avoid IP blocking
        time.sleep(sleep)

        # CACTUS url
        url = 'https://cactus.nci.nih.gov/chemical/structure/%s/smiles' % (cas)

        try:
            smiles = get(url).content.decode('utf-8')

            # If there is not SMILES for that CAS:
            if error in smiles:
                print ('[ERROR]\t No SMILES in CACTUS for:', cas)
                smiles_list.append('Unknown')

            else:
                smiles_list.append(smiles)
                n += 1

        # Handling other exceptions
        except Exception as e:
            print(e)

        # Limit the number of CAS processed in order to avoid IP blocking
        if limit and n > limit: break

    if len(smiles_list) == 1: smiles_list = smiles_list[0]


    return smiles_list


if __name__ == '__main__':

    train_data = pd.read_csv('CAS_sens.csv', header=0, encoding='utf-8')
    print((train_data['CASRN']).tolist())
    smiles_list = cas_to_smiles(train_data['CASRN'])
    print(smiles_list)
