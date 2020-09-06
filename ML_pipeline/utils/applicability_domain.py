SEED = 42

import numpy as np
np.random.seed(SEED)

from collections import Counter, defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import scipy.stats as ss
from statistics import median

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd


# TODO: save AD best parameters to JSON


class ADCalculator(object):
    '''Implementation of applicability domain calculation. Allows for Tanimoto
    similarity and descriptors distribution comparisons.
    '''
    def __init__(self, train_df, **kwargs):
        # Default class attributes
        self.fp_diam = 4
        self.n_bits = 1024
        self.desc_position = 1
        self.p_value = 0.05
        self.scaler = StandardScaler()
        # self.verbose = False

        # Update class attributes with keyword arguments
        self.__dict__.update(kwargs)

        self.train_desc = train_df.iloc[:10, self.desc_position:self.desc_position+5]
        # self.train_desc = train_df.iloc[:, self.desc_position:]
        self.train_desc = self._scale_continous(self.train_desc)

        self.train_smiles = train_df['SMILES'].values
        # self.train_mol = [Chem.MolFromSmiles(smi) for smi in self.train_smiles]
        self.train_mol = []

        for smi in self.train_smiles:
            self.train_mol.append(Chem.MolFromSmiles(smi))

        self.train_fps = [self._get_morgan(mol) for mol in self.train_mol]
        print('[+] Morgan fingerprints calculated')
        self.train_ad = self._get_ad()
        print('[+] Train AD calculated')


    def _get_morgan(self, mol):
        '''Auxiliary method for calculating Morgan fingerprints.
        '''
        try:
            return AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.fp_diam,
                nBits=self.n_bits
            )

        except:
            pass


    def _get_ad(self):
        '''Obtains the applicability domains for all the descriptors in the
        training set.
        '''
        ad_dict = dict()

        for desc in list(self.train_desc):
            desc_values = self.train_desc[desc].values.reshape(-1, 1)
            ad_dict[desc] = self._check_distribution(desc_values)

        return ad_dict


    def _check_distribution(self, desc_values):
        '''Checks distribution of a single descriptor in order to return
        a suitable method for determining the applicability domain for such
        descriptor.
        '''
        assert isinstance(desc_values, (np.ndarray))

        # Handle discrete variables (less than 10% unique values)
        if len(np.unique(desc_values)) < desc_values.shape[0]*0.1:
            proba_estimator = Counter(desc_values.flatten().tolist())
            for key in proba_estimator:
                proba_estimator[key] /= desc_values.shape[0]

            return proba_estimator

        # If uniform distribution, apply ranges method
        args = (np.amin(desc_values), np.amax(desc_values))
        stat, p_value = ss.kstest(desc_values, 'uniform', args=args)
        if p_value >= self.p_value:
            # for desc in self.train_desc.keys():
            #     print(desc, min(desc_values))
            raise NotImplementedError

        # If normal distribution, apply distances method
        stat, p_value = ss.kstest(desc_values, 'norm')
        if p_value >= self.p_value:
            # fit Minimum Covariance Determinant (MCD) robust estimator to data
            robust_cov = MinCovDet(support_fraction=1)

            return robust_cov.fit(desc_values)

        # Else (non-parametric distribution), apply KDE
        else:
            # WARNING: if discrete, inside the range is higher probability
            # (interpolation), which cannot be always true (some values
            # inside the range cannot be in the applicability domain?)

            # NOTE: no need to change distance to Manhattan because input is
            # assumed to be scaled

            hyperparams = {
                'bandwidth': np.linspace(0.5, 1.0, 20),
                'kernel': [
                    'gaussian',
                    'epanechnikov',
                    'exponential',
                    'linear'
                ]
            }

            grid = GridSearchCV(KernelDensity(), hyperparams, cv=10)
            grid.fit(desc_values)

            kde = grid.best_estimator_

            # print(grid.best_params_)
            #
            # x_grid = np.linspace(-4.5, 7.0, 1000).reshape(-1, 1)
            # pdf = np.exp(kde.score_samples(x_grid))
            #
            # fig, ax = plt.subplots()
            # ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
            # ax.hist(desc_values, 30, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
            # ax.legend(loc='upper left')
            # ax.set_xlim(-4.5, 3.5)
            # plt.show()

            return kde


    def _scale_continous(self, df, fitted=False):
        continous = list(df.select_dtypes(include=['float']).columns.values)

        # Return original dataframe if all variables are categorical
        if not len(continous):
            return df

        if not fitted:
            df[continous] = \
                self.scaler.fit_transform(df[continous])

        else:
            df[continous] = \
                self.scaler.transform(df[continous])

        return df


    def tanimoto(self, smiles, top_scores=10):
        '''Calculates Tanimoto similarity between query SMILES and training set.
        '''
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        fp = self._get_morgan(mol)

        results = []
        for fp_train in self.train_fps:
            try:
                res = round(DataStructs.FingerprintSimilarity(fp, fp_train), 2)
                results.append(res)

            # Morgan fingerprints not correctly calculated
            except:
                results.append(0.0)

        results = list(sorted(results))[::-1]

        return results[:top_scores]


    def check_ad(self, query_desc):
        '''Queries a set of molecules for their inclusion in the applicability
        domain of the training set.
        '''
        # TODO: change method to transform only
        # query_desc = self._scale_continous(query_desc)

        ad_dict = defaultdict(list)
        for idx, row in query_desc.iterrows():
            for desc in self.train_ad.keys():
                if self.train_ad[desc] is None:
                    print('Descriptor is None!')
                    # TODO: Avoid when methods (ranges and distances) are finished!
                    continue

                if desc not in query_desc.columns:
                    print(desc, 'Descriptor out!')
                    # Many Mordred desc are not present in DRAGON and viceversa
                    # TODO: handle when desc of the query is Inf, NaN, etc.
                    # raise NotImplementedError
                    continue

                desc_value = np.array(row[desc]).reshape(-1, 1)

                if isinstance(self.train_ad[desc], dict):
                    probability = self.train_ad[desc][desc_value]
                else:
                    log_pdf = self.train_ad[desc].score_samples(desc_value)
                    probability = np.exp(log_pdf)

                ad_dict[idx].append(probability)

        return [np.median(values) for values in ad_dict.values()]


    def check_residues(self):
        return
