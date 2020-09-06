
import pandas as pd
import numpy as np

SEED = 42

from numpy.random import seed
seed(SEED)

from sklearn.model_selection import train_test_split

from utils.json_files import load_json

from sklearn.model_selection import cross_val_score

from custom_model_selection.model_selector import SKModelSelector
from sklearn.metrics import *
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import  KBinsDiscretizer

from sklearn.model_selection import validation_curve

# Algorithms
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, Ridge, PassiveAggressiveRegressor, LassoLars
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neural_network import MLPRegressor
# import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from reportlab.platypus import Frame, Paragraph, Spacer, Image
from reportlab.lib.units import cm
import seaborn as sns
sns.set_style('white')



# print("Please, check that you have correctly set the following parameters for your prediction:")
# print(" - MODEL")
# print(" - TARGET_COL")
# print(" - dataset (file path)")
# print(" - selected_features")
# print(" - kwargs (algorithm)")
# print(" - if SMILES is the correct name column in the input dataset")
# print(" - test size")
#
# input("Sure that you want to continue?")



############################################################################
#########################    LOAD DATA     #########################
############################################################################

PATH = input('Please input your PATH (enter to: "../data/"): ') or "../data/"
MODEL = input('Please input your MODEL NAME (enter to: avian_reproduction_toxicity): ') or "avian_reproduction_toxicity"
METHOD = input('Please input your MODEL TYPE [RFE/GBM/PI](enter to: PI): ') or "PI"

TARGET_COL = 'y'

train_data = pd.read_csv('{}-train_reduction_{}.csv'.format(PATH+MODEL,METHOD), sep=';')
test_dataset = pd.read_csv('{}-test_reduction_{}.csv'.format(PATH+MODEL,METHOD), sep=';')

# dataset.tail()
###############################################################################

###### DESCOMENTAR CUANDO HAGA EL 1 ####

# print("HEAD OF YOUR DATASET:")
# print(dataset.head())
# print("SHAPE OF YOUR DATASET", dataset.shape)
# dataset.tail()
# dataset_features = dataset.iloc[:, 2:] # Here, you can filter by number of features

# dataset_features.head()
# dataset_features.shape


# dataset_labels = dataset[TARGET_COL]



# # Number of descriptors to be selected for the FeedForwardNet
# SELECT_K_VARS = 40

###############################################################################
############################# FUNCTION TO SHOW PLOTS ##########################
###############################################################################


def regression_plots(y_true, y_pred, set):

    residues = y_true - y_pred
    data = pd.DataFrame({
        'Observed': y_true,
        'Predicted': y_pred,
        'Residues': residues,
    })

    palette = sns.light_palette((210, 90, 60), as_cmap=True, input='husl')
    #
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, sharex=False)

    fig.set_size_inches(5, 2.5)
    fig.subplots_adjust(wspace=0.5, hspace=0, bottom = 0.2)

    scatter_kws={'s': 7, 'c': abs(data['Residues']), 'cmap': palette, 'color': None}


    ax1.scatter(
        x=data['Observed'], y=data['Predicted'], s=7, c=abs(data['Residues']), cmap=palette, color=None
    )

    # Add diagonal line (line of equality)
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax1.plot(lims, lims, '-r')

    ax1.set_xlim(lims)
    ax1.set_ylim(lims)

    # Residues vs Observed
    sns.residplot(
        x='Observed', y='Residues', data=data, ax=ax2, scatter_kws=scatter_kws
    )

    # Add axis labels
    label_size = 7

    ax1.set_xlabel('Observed', fontsize=label_size)
    ax1.set_ylabel('Predicted', fontsize=label_size)

    ax2.set_xlabel('Observed', fontsize=label_size)
    ax2.set_ylabel('Residues', fontsize=label_size)

    ax1.tick_params(labelsize=label_size)
    ax2.tick_params(labelsize=label_size)

    plt.show()


if __name__ == '__main__':


    ############################################################################
    #########################    NAME COLUMNS     ##############################
    ############################################################################
    import random


    selected_features = list(train_data.iloc[:,2:])

    print(selected_features)
    #
    # # selected_features.remove('MAXDP')
    # # selected_features.remove('MATS3m')



    # selected_features.append()


    # selected_features = ['PW2', 'TI1', 'nMultiple', 'MATS1v', 'JhetZ', 'MATS1d', 'C-040', 'AATSC0dv', 'AATS6p']

        # out = random.choice(selected_features)
    # selected_features.remove(out)
    #
    #
    #
    # print(out)

    # selected_features = []

    # for i in range(20):
    #     inn = random.choice(selected_features_0)
    #     selected_features.append(inn)
    #     selected_features_0.remove(inn)

######### DESCOMENTAR CUANDO HAGA EL 1 ####

    # train_selected = dataset_features[selected_features]
    # print("Selected_features = " , selected_features)
    # print("Number_selected_features = ", len(selected_features))



    # ############################################################################
    # ########################     ML model training      ########################
    # ############################################################################


    X_train = train_data.iloc[:, 2:]
    X_test = test_dataset.iloc[:, 2:]
    y_train = train_data['y']
    y_test = test_dataset['y']

    train_set = train_data.iloc[X_train.index.values]
    test_set = test_dataset.iloc[X_test.index.values]

    print("\nPARAMETERS")
    print('train_set', X_train.shape)
    print('test_set', X_test.shape, "\n")

    # Load hyperparams from JSON file
    hyperparams = load_json('custom_model_selection/hyperparams_regression.json')

    # Update hyperparams dict with random seed and parallelization
    no_random_state = [
        'LinearRegression', 'LDA', 'LassoLars', 'SGDRegressor',
        'SVR', 'NuSVR',
        'KNeighborsRegressor',
        'LGBMRegressor', 'XGBRegressor', 'StackedGeneralizer',
        'FeedForwardNet',

    ]
    no_n_jobs = [
        'LinearRegression', 'LDA', 'Lasso', 'LassoLars', 'SGDRegressor', 'Ridge',
        'SVC', 'SVR', 'NuSVR',
        'LinearSVC', 'LinearSVR',
        'DecisionTreeRegressor',
        'AdaBoostRegressor', 'GradientBoostingRegressor',
        'LGBMRegressor', 'XGBRegressor', 'StackedGeneralizer',
        'FeedForwardNet', 'GaussianProcessRegressor', 'MLPRegressor','PassiveAggressiveRegressor'
    ]

    for key in hyperparams.keys():
        if key not in no_random_state:
            hyperparams[key]['random_state'] = [SEED]
        if key not in no_n_jobs:
            hyperparams[key]['n_jobs'] = [-1]

    # Update hyperparams for the FeedForwardNet
    # hyperparams['FeedForwardNet']['input_dim'] = [SELECT_K_VARS]
    # hyperparams['FeedForwardNet']['validation_data'] = [(X_test, y_test)]

    kwargs = {
        'models_list': [
            ###### LINEALES
                (LinearRegression, hyperparams['LinearRegression']),
             #
             (Ridge, hyperparams['Ridge']),
             #(GaussianProcessRegressor, hyperparams['GaussianProcessRegressor']),
             #(PassiveAggressiveRegressor,hyperparams['PassiveAggressiveRegressor']),
             #(Lasso, hyperparams['Lasso']),
             #(LassoLars, hyperparams['LassoLars']),
             #(SGDRegressor, hyperparams['SGDRegressor']), ## CONFIGURE validation_fraction hyperparam!!!!
             #(SVR, hyperparams['SVR']),
             #(LinearSVR, hyperparams['LinearSVR']),
             #(NuSVR, hyperparams['NuSVR']),
            # # #   #### KNN
                # (KNeighborsRegressor, hyperparams['KNeighborsRegressor']),
            # # # #   ####Trees
                # (DecisionTreeRegressor, hyperparams['DecisionTreeRegressor']),
                # (RandomForestRegressor, hyperparams['Forests']),
                (ExtraTreesRegressor, hyperparams['Forests']),
                # (GradientBoostingRegressor, hyperparams['GradientBoostingRegressor']),
                # (lgb.LGBMRegressor, hyperparams['LGBMRegressor']),
            #
             # (MLPRegressor, hyperparams['MLPRegressor']),
            # ##### Ada and others
             #(AdaBoostRegressor, hyperparams['AdaBoostRegressor']),
            #(xgb.XGBRegressor,hyperparams['XGBRegressor']),
        ],
        'cv': 10,
        'scoring': 'r2'
    }

    # Feature scaling


###### DESCOMENTAR CUANDO HAGA EL 1 ######

    # # Save the test and training sets for the AD estimation in app and metrics
    # train_set = dataset.iloc[X_train.index.values]
    # train_set = train_set.loc[:, ['SMILES'] + list(selected_features) + [TARGET_COL]]
    # train_set.rename(columns={'y': 'observed'}, inplace=True)
    #
    # test_set = dataset.iloc[X_test.index.values]
    # test_set = test_set.loc[:, ['SMILES'] + list(selected_features) +  [TARGET_COL]]
    # test_set.rename(columns={'y': 'observed'}, inplace=True)



    X_train = train_data.iloc[:, 2:]
    X_test = test_dataset.iloc[:, 2:]
######################## in case you want to scale

    # from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

    # scaler = StandardScaler()

    # scaler.fit(X_train)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # y_train = np.array(y_train).reshape ((len(y_train), 1))

    # y_train = scaler.fit_transform(y_train)


    # y_test = np.array(y_test).reshape ((len(y_test), 1))

    # y_test = scaler.fit_transform(y_test)

    # print(min(y_test))









    ###########################

    print()
    print('[+] Training the model...')


    model = SKModelSelector(**kwargs).fit((X_train, y_train), (X_test, y_test))

    # test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
    # test_y = reg.predict(test_x)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(x, y, s=10, c='b', marker="s", label='real')
    # ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
    #
    # plt.show()
    ############################################################################
    #########################     ML model pickling     ########################
    ############################################################################

    import pickle
    pickle.dump(model, open('{}{}.sav'.format(PATH,MODEL), 'wb'))

    # print('X_test', X_test)
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    ####### DESCOMENTAR CUANDO HAGA EL 1 #####


    train_set = train_data.assign(predicted = y_train_pred)
    test_set = test_dataset.assign(predicted = y_test_pred)

    train_set.to_csv('{}{}-descriptors-train.txt'.format(PATH,MODEL), sep='\t', index=False)
    test_set.to_csv('{}{}-descriptors-test.txt'.format(PATH,MODEL), sep='\t', index=False)

    print()
    print('Train results R^2:\t', r2_score(y_train, y_train_pred))
    print('Test results R^2:\t', r2_score(y_test, y_test_pred))

    RSS_train = mean_squared_error(y_train,y_train_pred)
    print('\nTrain results MSE:\t' , RSS_train)
    RSS_test = mean_squared_error(y_test,y_test_pred)
    print('Test results MSE\t' , RSS_test)


    print('\t|Train | Test')
    print('    EV\t|%3.2f  |%3.2f' %(explained_variance_score(y_train,y_train_pred), explained_variance_score(y_test,y_test_pred)))
    print('   MAE\t|%3.2f  |%3.2f' %(mean_absolute_error(y_train,y_train_pred), mean_absolute_error(y_test,y_test_pred)))
    print('   MSE\t|%3.2f  |%3.2f' %(mean_squared_error(y_train,y_train_pred), mean_squared_error(y_test,y_test_pred)))
    print(' MEDAE\t|%3.2f  |%3.2f' %(median_absolute_error(y_train,y_train_pred), median_absolute_error(y_test,y_test_pred)))
    print('Rscore\t|%3.2f  |%3.2f' %(r2_score(y_train,y_train_pred), r2_score(y_test,y_test_pred)))



    # CV SCORES
    # explained_variance_score = make_scorer(explained_variance_score)
    # mean_absolute_error = make_scorer(mean_absolute_error)
    # mean_squared_error = make_scorer(mean_squared_error)
    # median_absolute_error = make_scorer(median_absolute_error)
    # r2_score = make_scorer(r2_score)
    #
    # evs_train = cross_val_score(model, X_train, y_train, cv = 5, scoring = explained_variance_score)
    # mae_cv_train = cross_val_score(model, X_train, y_train, cv = 5, scoring = mean_absolute_error)
    # mse_cv_train = cross_val_score(model, X_train, y_train, cv = 5, scoring = mean_squared_error)
    # medae_cv_train = cross_val_score(model, X_train, y_train, cv = 5, scoring = median_absolute_error)
    # r2_cv_train = cross_val_score(model, X_train, y_train, cv = 5, scoring = r2_score)
    #
    #
    # evs_train_final = np.mean(evs_train)
    # mae_cv_train_final = np.mean(mae_cv_train)
    # mse_cv_train_final = np.mean(mse_cv_train)
    # medae_cv_train_final = np.mean(medae_cv_train)
    # r2_cv_train_final = np.mean(r2_cv_train)
    #
    # evs_test = cross_val_score(model, X_test, y_test, cv = 5, scoring = explained_variance_score)
    # mae_cv_test = cross_val_score(model, X_test, y_test, cv = 5, scoring = mean_absolute_error)
    # mse_cv_test = cross_val_score(model, X_test, y_test, cv = 5, scoring = mean_squared_error)
    # medae_cv_test = cross_val_score(model, X_test, y_test, cv = 5, scoring = median_absolute_error)
    # r2_cv_test = cross_val_score(model, X_test, y_test, cv = 5, scoring = r2_score)
    #
    #
    # evs_test_final = np.mean(evs_test)
    # mae_cv_test_final = np.mean(mae_cv_test)
    # mse_cv_test_final = np.mean(mse_cv_test)
    # medae_cv_test_final = np.mean(medae_cv_test)
    # r2_cv_test_final = np.mean(r2_cv_test)
    #
    #
    # print('\n')
    # print("Cross-validation results:")
    #
    #
    #
    # print('\t|Train | Test')
    # print('    EV\t|%3.2f  |%3.2f' %(evs_train_final,evs_test_final))
    # print('   MAE\t|%3.2f  |%3.2f' %(mae_cv_train_final,mae_cv_test_final))
    # print('   MSE\t|%3.2f  |%3.2f' %(mse_cv_train_final,mse_cv_test_final))
    # print(' MEDAE\t|%3.2f  |%3.2f' %(medae_cv_train_final,medae_cv_test_final))
    # print('Rscore\t|%3.2f  |%3.2f' %(r2_cv_train_final,r2_cv_test_final))


    #PLOT prediction

    fig, axs = plt.subplots(2)
    x_ax_train=range(np.shape(y_train)[0])
    axs[0].scatter(x_ax_train, y_train, s=5, color="blue", label="original")
    axs[0].plot(x_ax_train, y_train_pred, lw=1.5, color="red", label="predicted")
    axs[0].legend()

    x_ax_test=range(np.shape(y_test)[0])
    axs[1].scatter(x_ax_test, y_test, s=5, color="blue", label="original")
    axs[1].plot(x_ax_test, y_test_pred, lw=1.5, color="red", label="predicted")
    plt.legend()
    plt.show()

    regression_plots(y_train, y_train_pred, 'train')
    regression_plots(y_test,y_test_pred, 'test')
