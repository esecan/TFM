from rdkit import Chem

path = 'data/Toxicology/skin_irr/datasets/'

sdf = Chem.SDMolSupplier(path + 'RNB_skin_irrit.sdf')

with open(path + 'RNB_skin_irrit.txt', 'w') as f:
    for mol in sdf:
        try:
            smi = Chem.MolToSmiles(mol)
            print(smi)
            f.write("{}\n".format(smi))

        except:
            continue
