# -*- coding: utf-8 -*-
'''
Creators: Joel Roca and updated by Eva Serrano
Date of update: 08/2019
'''

###############################################################################
#################################### IMPORTS ##################################
###############################################################################

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel

from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier
    )

from eli5.sklearn import PermutationImportance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from modules.feature_selector import FeatureSelector
from modules import split_by_kmeans

###############################################################################
################################### FUNCTIONS  ################################
###############################################################################


################################# main menu ####################################
def main_menu():

    print('######################### MAIN MENU #########################')
    print('\nPlease select what do you want to do: ')

    print('[1] Initial feature reduction: infinite, correlated, constant and empty values')
    print('[2] Generation of train and test sets based in kmeans')
    print('[3] Feature selection by RFE')
    print('[4] Feature selection by LGBM')
    print('[5] Feature selection by Permutation importance')
    print('[6] Exit NEO')

    flag_menu = True

    allowed = ['1','2','3','4','5','6']

    main_menu_choice = input('\nYour choice: ')

    while flag_menu:

        if main_menu_choice in allowed:
            flag_menu = False
        else:
            main_menu_choice = input('\nIncorrect input. Your choice: ')
            continue

    if main_menu_choice == '6':
        print('\nThanks for using NEO!')
        exit()

    return main_menu_choice

################################ read dataframe ################################
def read_dataframe(INPUT_FILE):

    dataset = pd.read_csv(
        PATH + INPUT_FILE,
        sep = SEP,
        header=0,
        encoding='latin'
        )

    return dataset

################################ file checkpoint ###############################

def file_checkpoint():
    flag_fc = True

    allowed_fc = ['y','Y','n','N']

    file_checkpoint = input("Continue (Y/n)?") or "y"
    while flag_fc:
        if file_checkpoint in allowed_fc:
            flag_fc = False
        else:
            file_checkpoint = input('\nIncorrect input. Continue (Y/n)?' or "y")
            continue

    if file_checkpoint == 'n' or file_checkpoint == 'N':
        main()

###############################################################################
################################  Set params ##################################
###############################################################################

SEED = 42
TARGET_COL = 'y'
SEP = ';'
TEST_SIZE = 0.25

print('\n#########################################################################'
        + '\n######################### WELCOME TO NEO script #########################'
        + '\n#########################################################################'
        + '\nThis script will allow you to: \n'
        + ' \t- perform the train/test split based on kmeans\n'
        + ' \t- select the relevant features based on:\n'
        + '\t\t · Recursive feature elimination (RFE)\n'
        + '\t\t · Ligth gradient boosting machine (LGBM)\n'
        + '\t\t · Permutation importance (PI)\n')

PATH = input('Please input your PATH (enter to: "../data/"): ') or "../data/"
MODEL = input('Please input your MODEL NAME (enter to: avian_reproduction_toxicity): ') or "avian_reproduction_toxicity"
TEST_SIZE = input('Please input your desired TEST SIZE (enter to: "0.25"): ') or "0.25"
TEST_SIZE = float(TEST_SIZE)

def main():

    main_menu_choice = main_menu()

    ###########################################################################
    ####################### Initial Feature reduction #########################
    ###########################################################################

    if main_menu_choice == '1':

        ############################ Load file ################################
        print('A file located in "{}" folder is needed'.format(PATH))
        print('This file must be called: "{}-paralel_calculated_with_y.csv"'.format(MODEL))
        file_checkpoint()

        INPUT_FILE = '{}-paralel_calculated_with_y.csv'.format(MODEL)
        dataset = read_dataframe(INPUT_FILE)

        #######################################################################

        dataset_features = dataset.iloc[:, 2:]

        # set multiclass criteria
        multiclass = False
        if multiclass:
            dataset_labels = dataset[TARGET_COL].apply(str)

        else:
            dataset_labels = dataset[TARGET_COL]


        SMILES = dataset['SMILES']

    #    start = time.time()

        # print()
        print('[1] Initial feature reduction: infinite, correlated, constant and empty values')
        # print()
        #
        # Identify columns with values too large
        cols = (dataset_features > 1e38).any()
        print(sum(cols), 'infinite values') # must be zero

        if sum(cols) >0:
            kk = list(cols)
            todrop=[]
            for i,k in enumerate(kk):
                if k == True:
                    print(k)
                    todrop.append(i)

            dataset_features = dataset_features.drop(dataset_features.columns[todrop], axis=1)

        dataset_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Instance the feature selector class
        fs = FeatureSelector(data = dataset_features, labels = dataset_labels)

        # 1. Identify descriptors with a percentage of missing values > threshold
        fs.identify_missing(missing_threshold = 0.00000000001)
        missing_features = fs.ops['missing']
        # fs.plot_missing()

        # 2. Identify collinear features and keep one of them and remove the rest  (Deterministic)
        fs.identify_collinear(correlation_threshold = 0.9)
        collinear_features = fs.ops['collinear']
        # fs.plot_collinear()

        # 3. Identify descriptors that have a unique value
        fs.identify_single_unique()

        # fs.plot_unique()


        # Remove features based on previous

        train_removed = fs.remove(methods = ['missing', 'collinear', 'single_unique'],
                                  keep_one_hot=False)




        train_removed.shape
        train_removed.head()
    #    print("hello")
    #    end = time.time()
    #    print('total', end - start)
        # Save initial_reduction
        initial_red = train_removed.copy (deep=True)
        initial_red.insert(0,'SMILES',SMILES)
        initial_red.insert(1,'y',dataset_labels)

        initial_red.shape

        initial_red.tail()

        initial_red.to_csv('{}-initial_reduction.csv'.format(PATH+MODEL), sep=';', index=False)

        print("\nThe following files have been created:\n")
        print('{}-initial_reduction.csv'.format(PATH+MODEL))

        ############################ back to main menu ########################
        confirm = input("\nDo you want to perform any other step?(y/n):  ") or "y"

        if confirm == 'y' or confirm == 'Y':

            main()

        else:
            print('\nThanks for using NEO!')
            exit()
        ###############################################################################




#%%
    ###########################################################################
    #################### Generation of train and test sets   ##################
    ###########################################################################
    if main_menu_choice == '2':

        ############################ Load file ################################
        print('A file located in "{}" folder is needed'.format(PATH))
        print('This file must be called: "{}-initial_reduction.csv"'.format(MODEL))
        file_checkpoint()

        INPUT_FILE = '{}-initial_reduction.csv'.format(MODEL)
        initial_red = read_dataframe(INPUT_FILE)

        #######################################################################

        print('[+] Generation of train and test sets based in kmeans')

        data_df,train_set, test_set = split_by_kmeans.create_clusters(initial_red, SEED, TEST_SIZE,TARGET_COL)


        if data_df.shape != initial_red.shape:

            data_df.to_csv('{}-initial_reduction_cleaned_from_kmeans.csv'.format(PATH+MODEL), sep=';', index=False)
            train_set.to_csv('{}-train_set.csv'.format(PATH+MODEL), sep=';', index=False)
            test_set.to_csv('{}-test_set.csv'.format(PATH+MODEL), sep=';', index=False)

            print("\nThe following files have been created:\n")
            print('{}-cleaned_from_kmeans.csv'.format(PATH+MODEL))
            print('{}-train_set.csv'.format(PATH+MODEL))
            print('{}-test_set.csv'.format(PATH+MODEL))

        else:

            train_set.to_csv('{}-train_set.csv'.format(PATH+MODEL), sep=';', index=False)
            test_set.to_csv('{}-test_set.csv'.format(PATH+MODEL), sep=';', index=False)

            print("\nThe following files have been created:\n")
            print('{}-train_set.csv'.format(PATH+MODEL))
            print('{}-test_set.csv'.format(PATH+MODEL))

        ############################ back to main menu ########################
        confirm = input("\nDo you want to perform any other step?(y/n):  ") or "y"

        if confirm == 'y' or confirm == 'Y':

            main()

        else:
            print('\nThanks for using NEO!')
            exit()
        ###############################################################################

    ###########################################################################
    ###########################    Reduction by RFE     #######################
    ###########################################################################

    if main_menu_choice == '3':
        print('[+] Reduction by RFE')

        ############################ Load file ################################
        print('Two files located in "{}" folder is needed'.format(PATH))
        print('These files must be called:'
                + '\n\t"{}-train_set.csv"'.format(MODEL)
                + '\n\t"{}-test_set.csv"'.format(MODEL))
        file_checkpoint()

        INPUT_FILE = '{}-train_set.csv'.format(MODEL)
        dataset = read_dataframe(INPUT_FILE)

        INPUT_FILE_test = '{}-test_set.csv'.format(MODEL)
        dataset_test = read_dataframe(INPUT_FILE_test)

        #######################################################################


        train_removed = dataset.iloc[:, 2:]
        SMILES = dataset['SMILES']

        multiclass = False
        if multiclass:
            dataset_labels = dataset[TARGET_COL].apply(str)

        else:
            dataset_labels = dataset[TARGET_COL]


        ####################### menu RFE estimator ############################

        print("\nPlease select your estimator for perform RFE: ")

        print('[1] SVR with linear kernel')
        print('[2] Ridge')
        print('[3] LinearRegression')


        menu_RFE_estimator_choice = input('Your choice: ')

        if menu_RFE_estimator_choice == '1':

            print("\nPlease define your parameters for SVR with linear kernel estimator: ")
            c = int(input('C:'))
            estimator = SVR(kernel="linear", C = c)

        elif menu_RFE_estimator_choice == '2':

            print("\nPlease define your parameters for Ridge regression estimator: ")
            alpha = float(input('alpha:'))
            estimator = linear_model.Ridge(alpha=alpha)

        elif menu_RFE_estimator_choice == '3':

           estimator = linear_model.LinearRegression()


        #######################################################################

        X = np.array(train_removed)
        y = np.array(dataset['y'])

        ########################## Define number of features ##################
        print('\nNow you can select a number of features and steps.')
        flag = True

        while flag:


            SELECT_K_VARS = int(input('Please, indicate the number of selected features: '))
            steps = int(input('Please, indicate the number of steps:'))

            selector = RFE(estimator, SELECT_K_VARS, step = steps)
            print('Please wait...')
            selector = selector.fit(X, y)

            pred_y = selector.predict(X)

            np.shape(y)[0]
            # prediction scores
            RSS = mean_squared_error(y,selector.predict(X)) * len(y)
            RSS
            R_squared = selector.score(X,y)

            # filter train dataset with selected descriptors:
            feat = list(selector.get_support())
            s = pd.Series(feat)

            selected_ln = train_removed[train_removed.columns[s]]
            features = list(selected_ln.columns)

            print('\nSelected features:')
            print(features)
            print('R_squared', R_squared)

            selected_ln.insert(0,'SMILES',SMILES)
            selected_ln.insert(1,'y',dataset_labels)
            selected_ln.to_csv('{}-train_reduction_RFE.csv'.format(PATH+MODEL), sep=';', index=False)

            # filter test dataset with selected descriptors:
            list_columns2 = list(selected_ln)
            dataset_test_filtered = dataset_test[list_columns2]
            dataset_test_filtered.to_csv('{}-test_reduction_RFE.csv'.format(PATH+MODEL), sep=';', index=False)

            print("\nThe following files have been created:\n")
            print('{}-train_reduction_RFE.csv'.format(PATH+MODEL))
            print('{}-test_reduction_RFE.csv'.format(PATH+MODEL))

            print('train_set', selected_ln.shape)
            print('test_set', dataset_test_filtered.shape, "\n")


            continue_select_K_VARS = input("\nDo you agree with that number of features and steps?(y/n): ")

            if continue_select_K_VARS == 'y' or continue_select_K_VARS=='Y':
                flag = False
                pass
            else:
                flag = True
                pass
        #######################################################################


        ############################ back to main menu ########################
        confirm = input("\nDo you want to perform any other step?(y/n):  ") or "y"

        if confirm == 'y' or confirm == 'Y':

            main()

        else:
            print('\nThanks for using NEO!')
            exit()
        ###############################################################################

    ###########################################################################
    ########################### Reduction by lgbm #############################
    ###########################################################################
    if main_menu_choice == '4':
        print('[+] Reduction by lgbm')

        ############################ Load file ################################
        print('Two files located in "{}" folder is needed'.format(PATH))
        print('These files must be called:'
                + '\n\t"{}-train_set.csv"'.format(MODEL)
                + '\n\t"{}-test_set.csv"'.format(MODEL))
        file_checkpoint()

        INPUT_FILE = '{}-train_set.csv'.format(MODEL)
        dataset = read_dataframe(INPUT_FILE)

        INPUT_FILE_test = '{}-test_set.csv'.format(MODEL)
        dataset_test = read_dataframe(INPUT_FILE_test)

        #######################################################################


        train_removed = dataset.iloc[:, 2:]
        SMILES = dataset['SMILES']

        multiclass = False
        if multiclass:
            dataset_labels = dataset[TARGET_COL].apply(str)

        else:
            dataset_labels = dataset[TARGET_COL]

        ########################## Define model type ##########################

        model_type = input('\nPlease define if your model is for [1] classification or [2] regression:')

        if model_type == '1':
            task = 'classification'
            print("\nPlease define your parameters for lgbm selection for classification parameters: ")
            eval_metric = input('eval_metric (l2/auc/binary_logloss):')

        if model_type == '2':

            task = 'regression'
            print("\nPlease define your parameters for lgbm selection for classification parameters: ")
            eval_metric = input('eval_metric (l2/rmse/l1):')

        #######################################################################
        fs_2 = FeatureSelector(data = train_removed, labels = dataset_labels)


        fs_2.identify_zero_importance(task = task,
                                      eval_metric = eval_metric,
                                      n_iterations = 10,
                                      early_stopping = True)


        zero_importance_features = fs_2.ops['zero_importance']
        fs_2.plot_feature_importances(threshold = 0.90, plot_n = 20)

        # 2.2. Identify features with low importance (Stochastic)
        fs_2.identify_low_importance(cumulative_importance = 0.90)

        fs_2.feature_importances

        print(fs_2.feature_importances)

        fs_2.feature_importances.to_csv('{}-train_featured_importances.csv'.format(PATH+MODEL), sep=';', index=False)

        print("\nThe following files have been created:\n")
        print('{}-train_featured_importances.csv'.format(PATH+MODEL))

        ########################## Define number of features ##################
        print('\nNow you can select a number of features.')
        flag = True

        while flag:


            SELECT_K_VARS = int(input('Please, indicate the number of selected features: '))

            selected_features = fs_2.feature_importances.head(SELECT_K_VARS)
            # selected_features = INPUT_DS_TOCHECK.head(SELECT_K_VARS)
            selected_features = selected_features['feature'].tolist()
            print('\nSelected features: \n', selected_features)

            # filter train dataset with selected descriptors:
            selected__GB = dataset[selected_features]
            selected__GB.insert(0,'SMILES',SMILES)
            selected__GB.insert(1,'y',dataset_labels)
            selected__GB.to_csv('{}-train_reduction_GBM.csv'.format(PATH+MODEL), sep=';', index=False)

            # filter test dataset with selected descriptors:
            list_columns = list(selected__GB)
            dataset_test_filtered = dataset_test[list_columns]
            dataset_test_filtered.to_csv('{}-test_reduction_GBM.csv'.format(PATH+MODEL), sep=';', index=False)

            print("\nThe following files have been created:\n")
            print('{}-train_reduction_GBM.csv'.format(PATH+MODEL))
            print('{}-test_reduction_GBM.csv'.format(PATH+MODEL))

            print('train_set', selected__GB.shape)
            print('test_set', dataset_test_filtered.shape, "\n")


            continue_select_K_VARS = input("\nDo you agree with that number of features?(y/n): ")

            if continue_select_K_VARS == 'y' or continue_select_K_VARS=='Y':
                flag = False
                pass
            else:
                flag = True
                pass
        #######################################################################

        ############################ back to main menu ########################
        confirm = input("\nDo you want to perform any other step?(y/n):  ") or "y"

        if confirm == 'y' or confirm == 'Y':

            main()

        else:
            print('\nThanks for using NEO!')
            exit()
        #######################################################################

    ###########################################################################
    #################    Reduction by Permutation importance     ##############
    ###########################################################################

    if main_menu_choice == '5':
        print('[+] Feature selection by Permutation importance')

        ############################ Load file ################################
        print('Two files located in "{}" folder is needed'.format(PATH))
        print('These files must be called:'
                + '\n\t"{}-train_set.csv"'.format(MODEL)
                + '\n\t"{}-test_set.csv"'.format(MODEL))
        file_checkpoint()

        INPUT_FILE = '{}-train_set.csv'.format(MODEL)
        dataset = read_dataframe(INPUT_FILE)

        INPUT_FILE_test = '{}-test_set.csv'.format(MODEL)
        dataset_test = read_dataframe(INPUT_FILE_test)

        #######################################################################

        train_removed = dataset.iloc[:, 2:]
        SMILES = dataset['SMILES']

        multiclass = False
        if multiclass:
            dataset_labels = dataset[TARGET_COL].apply(str)

        else:
            dataset_labels = dataset[TARGET_COL]

        ####################### menu Permutation importance ###################

        print("\nPlease select your estimator for perform Permutation importance: ")

        print('[1] ExtraTreesRegressor')
        print('[2] DecisionTreeRegressor')
        print('[3] GradientBoostingRegressor')
        print('[4] MLPRegressor')
        print('[5] RandomForestRegressor')
        print('[6] KNeighborsRegressor')
        print('[7] RandomForestClassifier')

        menu_PI_estimator_choice = input('Your choice: ')

        if menu_PI_estimator_choice == '1':
            estimator = ExtraTreesRegressor(bootstrap = True)

        elif menu_PI_estimator_choice == '2':
            estimator = DecisionTreeRegressor()

        elif menu_PI_estimator_choice == '3':
            estimator = GradientBoostingRegressor()

        elif menu_PI_estimator_choice == '4':
            estimator = MLPRegressor()

        elif menu_PI_estimator_choice == '5':
           estimator = RandomForestRegressor()

        elif menu_PI_estimator_choice == '6':
            estimator = KNeighborsRegressor(n_neighbors=2)

        elif menu_PI_estimator_choice == '7':
            estimator = RandomForestClassifier()

        print('Please wait...')

        #######################################################################
        X = np.array(train_removed)
        y = np.array(dataset['y'])

        estimator.fit(X, y)
        perm = PermutationImportance(estimator).fit(X, y)

        # perm.feature_importances_ attribute is now available, it can be used
        # for feature selection - let's e.g. select features which increase
        # accuracy by at least 0.01:
        sel = SelectFromModel(perm, threshold=0.01, prefit=True)

        perm.feature_importances_

        importances_ = list(perm.feature_importances_)
        features__permutation = list(train_removed.columns)


        dataframe_permutation = pd.DataFrame({'feature' : features__permutation,
                                    'importance' : importances_},
                                    columns=['feature','importance'])

        dataframe_permutation_ordered = dataframe_permutation.sort_values('importance',ascending=False)

        #######################################################################


        ########################## Define number of features ##################
        print('\nNow you can select a number of features.')
        flag = True

        while flag:


            SELECT_K_VARS = int(input('Please, indicate the number of selected features: '))

            most_important_permutation = dataframe_permutation_ordered.iloc[0:SELECT_K_VARS,0:1]

            features_selected_by_permutation = list(most_important_permutation['feature'])
            print('\nSelected features:\n', features_selected_by_permutation)

            # filter train dataset with selected descriptors:
            selected_by_permutation = dataset[features_selected_by_permutation]
            selected_by_permutation.insert(0,'SMILES',SMILES)
            selected_by_permutation.insert(1,'y',dataset_labels)
            selected_by_permutation.to_csv('{}-train_reduction_PI.csv'.format(PATH+MODEL), sep=';', index=False)


            # filter test dataset with selected descriptors:
            list_columns_by_permutation = list(selected_by_permutation)
            dataset_test_filtered = dataset_test[list_columns_by_permutation]
            dataset_test_filtered.to_csv('{}-test_reduction_PI.csv'.format(PATH+MODEL), sep=';', index=False)

            print("\nThe following files have been created:\n")
            print('{}-train_reduction_PI.csv'.format(PATH+MODEL))
            print('{}-test_reduction_PI.csv'.format(PATH+MODEL))

            print('train_set', selected_by_permutation.shape)
            print('test_set', dataset_test_filtered.shape, "\n")

            continue_select_K_VARS = input("\nDo you agree with that number of features?(y/n): ")

            if continue_select_K_VARS == 'y' or continue_select_K_VARS=='Y':
                flag = False
                pass
            else:
                flag = True
                pass
        #######################################################################

        ############################ back to main menu ########################
        confirm = input("\nDo you want to perform any other step?(y/n):  ") or "y"

        if confirm == 'y' or confirm == 'Y':

            main()

        else:
            print('\nThanks for using NEO!')
            exit()
        #######################################################################
#%%


if __name__ == "__main__":


    main()
