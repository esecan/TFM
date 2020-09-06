SEED = 42

import numpy as np
np.random.seed(SEED)

import re

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier


class SKModelSelector(object):
    '''General class for training and selecting the best scikit-learn models.
    '''
    def __init__(self, **kwargs):
        self.verbose = False

        # Update class attributes with keyword arguments
        self.__dict__.update(kwargs)

        self.scorer = metrics.SCORERS[self.scoring]


    def grid_search(self, model, hyperparams, X_train, y_train, X_test, y_test):
        '''Performs grid search for a single model.
        '''
        model_name = model.__name__
        # from sklearn.model_selection import train_test_split
        # train_features, valid_features, train_labels, valid_labels = train_test_split(X_train, y_train, test_size = 0.15)
        # Notes on RandomizedSearchCV: https://stats.stackexchange.com/questions/160479/practical-hyperparameter-optimization-random-vs-grid-search
        model = GridSearchCV(
            model(),
            # model(class_weight = {0:2.2,1:1}), ## oju: for weigthed balAnce: model(class_weight = {0:10,1:1})
            param_grid=hyperparams,
            # param_distributions=hyperparams,
            cv=self.cv,
            # n_iter=60,
            n_jobs=-1,
            scoring=self.scoring,
            iid=False

        )
        '''
        to analyze class_weigth matrix
        '''

        # from sklearn.utils import class_weight
        # class_weight = class_weight.compute_class_weight('balanced_subsample',
        #                                          np.unique(y_train),
        #                                          y_train)

        '''
        '''
        model.fit(X_train, y_train)

        # model.fit(X_train, y_train, eval_metric="l2",early_stopping_rounds=100, eval_set = [(valid_features, valid_labels)])

        print('\nBest parameters for {}:'.format(model_name))
        print(model.best_params_)

        print('\nGrid score:', model.best_score_)

        test_score = self.scorer(model, X_test, y_test)

        print('Test score:', test_score)

        # Update best score
        model.best_score_ = test_score

        if self.verbose:
            print('\nGrid scores on development set:')
            means = model.cv_results_['mean_test_score']
            stds = model.cv_results_['std_test_score']
            for mean, std, parameters in \
                zip(means, stds, model.cv_results_['params']):
                print('%0.3f (+/-%0.03f) for %r'
                      % (mean, std * 2, parameters))

        return model


    def fit(self, train, test):
        (X_train, y_train), (X_test, y_test) = train, test

        # TODO: change name to fit and transform
        # See: https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
        '''Selects one model among many, after performing grid search for each
        one of them separately.
        '''

        best_score = None
        best_model = None
        for model, hyperparams in self.models_list:
            grid_model = self.grid_search(
                model, hyperparams, X_train, y_train, X_test, y_test
            )

            # Initialize best_score and best_model
            if not best_score:
                best_score = grid_model.best_score_
                best_model = grid_model

            elif grid_model.best_score_ > best_score:
                best_model = grid_model
                best_score = grid_model.best_score_

            if self.verbose:
                print('Model:', grid_model)
                print('Score:', grid_model.best_score_)

        # Needed for prediction
        self.best_model = best_model

        print('\nBest model and parameters:')
        print(best_model)

        return best_model


    def predict(self, X_test):
        return self.best_model.predict(X_test)
