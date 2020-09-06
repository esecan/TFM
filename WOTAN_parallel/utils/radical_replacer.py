# -*- coding: utf-8 -*-

import re

# RDKit is an open-source Cheminformatics and Machine Learning library in Python
# Allows us to handle chemical data (molecule representations and so on)
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Mol


class RadicalReplacer(object):
    '''Radical Replacer (Sonrie) class for implementing SMILES templates
    substitution by different radicals.
    '''
    def __init__(self, data_dict, brackets=True):
        # Determine if the template radicals (Rs) are with or without brackets:
        # R1, R2, ... vs [R1], [R2], ...
        self.brackets = brackets

        # JSON structure:
        # one field for the templates (SMILES to be substituted)
        self.raw_templates = data_dict['Templates']
        # another one for the restricted radicals (not to be substituted)
        self.restricted = data_dict['Restricted']
        # another one for the substutients to incorporate into the templates
        self.substituents = data_dict['Substituents']

        # Only halogens checked
        self.atom_nums = [9, 17, 35, 53]

        # TODO: call as a method, not in __init__
        self.templates = sorted(self.replace_radicals(self.raw_templates))

        # TODO: Print/check undrawable molecules
        self.results = self.check_smiles_valence(self.templates)


    def count_radicals(self, temp, subs_letter='R'):
        '''Method for counting radicals in the templates to be substituted.

        Keyword arguments:
            temp (string) -- template
            subs_letter (string) -- letter of the radical to be substituted

        Output:
            radicals_list (list) -- list of radicals found in the template

        Observations:
            It could be useful to return the list of positions
        '''
        # Get the positions for all the radicals letters in the template
        positions = [i for i, letter in enumerate(temp)
                     if letter == subs_letter]

        # Get the radical and its number (R1, ...) if NOT in RESTRICTED
        radicals_list = [temp[pos:pos+2] for pos in positions
                        if temp[pos:pos+2] not in self.restricted]

        # If the radicals are between brackets:
        if self.brackets:
            # Radical = R + number + brackets ([R1], ...)
            radicals_list = ['[%s]' % rad for rad in radicals_list]

        return radicals_list


    def replace_radicals(self, templates):
        '''Method for replacing all the radicals in a given list of templates.

        Keyword arguments:
            templates (list) -- list of templates to be substituted

        Output:
            templates (list) -- list of templates with radicals substituted

        Observations:
            Function called in constructor method __init__
            Recursive function
        '''
        for temp in templates:
            # WARNING: temp is NOT A LIST!
            radicals = self.count_radicals(temp)

            if radicals:
                # Eliminate the current template if it contains R
                templates[templates.index(temp)] = None
                templates = list(filter(None, templates))

                # Replace the template with the possible substituents
                # WARNING: temp is a LIST!
                temp = [temp.replace(rad, sub)
                        for rad in radicals
                        for sub in self.substituents]
                templates.extend(self.replace_radicals(temp))

        return list(set(templates))


    def check_smiles_valence(self, smiles_list):
        '''Method for checking SMILES valence, avoiding aberrant replacements
        (not all substitutions are chemically logical).

        Keyword arguments:
            smiles_list (list) -- list of SMILES to be checked

        Output:
            list of MOLs with valences checked (aberrants eliminated)

        Observations:
            Function called in constructor method __init__
        '''
        mol_list = []
        for smi in smiles_list:
            # Avoid checking when the SMILES contains not substituted radicals:
            if re.findall('R\d+', smi):
                continue

            # Get the MOL from the SMILES
            mol = Chem.MolFromSmiles(smi)

            # If successfully converted to MOL:
            if mol:
                # Get the atom numbers from the MOL
                for atom in mol.GetAtoms():
                    # Check if the atom number is in the restricted ones
                    # (in our case, only halogens are taken into account) and
                    # check if their valence is different from 1 (aberrant)
                    if atom.GetAtomicNum() in self.atom_nums \
                    and atom.GetExplicitValence() != 1:
                        mol = None
                mol_list.append(mol)

        # Return filtered list, without aberrants
        return list(filter(None, mol_list))


if __name__ == '__main__':
    # EXAMPLE:
    # Load JSON as python dict
    import json
    with open('example.json', 'r', encoding='utf8') as json_file:
        data_dict = json.load(json_file)

    # Instantiate the Radical Replacer: replacement is done in the constructor
    # (__init__) method, therefore no extra steps are required
    replacer = RadicalReplacer(data_dict)

    # NOTE: warnings are generated during the step of valence check
    # (check_smiles_valence). These are messages indicating that the molecules
    # checked are aberrant and, so, are not chemically logical. They are not
    # incorporated into results, so no extra steps are required: the results
    # attribute of the RadicalReplacer instance contains ONLY checked and
    # feasible structures:
    for mol in replacer.results:
        print(Chem.MolToSmiles(mol))

    exit(0)

    # Save images for each result: execute only if needed!
    # For integration in a pipeline, check the MolToImage method from RDKit
    from rdkit.Chem import Draw
    n = 0
    for mol in replacer.results:
        n += 1
        Draw.MolToFile(mol, 'test%i.png' % n)
