# -*- coding: utf-8 -*-

import numpy as np

from sklearn.decomposition import PCA

def PCA_analysis(X, y, n_features='default'):
    if n_features == 'default': n_features = int(X.shape[0]/5)
    pca = PCA(n_components=n_features)
    X_reduced = pca.fit_transform(X)

    # for elem in pca.explained_variance_ratio_:
    #     print('{0:.9f}'.format(float(elem)))

    return X_reduced
