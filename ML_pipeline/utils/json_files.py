# -*- coding: utf-8 -*-

import json


def load_json(json_path):
    '''Loads JSON files as a dictionary.
    '''
    with open(json_path, 'r', encoding='utf8') as json_file:
        json_dict = json.load(json_file)

    return json_dict
