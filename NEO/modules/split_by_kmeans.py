# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:46:02 2020

@author: ProtoQSAR
"""

'''
Script to split the cleaned dataset in train and test set based in Kmeans
'''

import pandas as pd
import sys
current_module = module = sys.modules[__name__]

from math import *
import matplotlib.pyplot as plt
from scipy.spatial import distance


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def split_cluster(cluster,SEED, TEST_SIZE,TARGET_COL):

    dataset_features = cluster.iloc[:, 2:-1]

    dataset_labels = cluster[TARGET_COL]

    selected_features = list(cluster.iloc[:, 2:-1])

    train_selected = dataset_features[selected_features]

    # split

    X_train, X_test, y_train, y_test = train_test_split(
        train_selected, dataset_labels, test_size=TEST_SIZE, random_state=SEED
    )

    train_set = cluster.loc[X_train.index.values]
    train_set = train_set.loc[:, ['SMILES'] + [TARGET_COL] + list(selected_features)]
    train_set.head()

    test_set = cluster.loc[X_test.index.values]
    test_set = test_set.loc[:, ['SMILES'] + [TARGET_COL] + list(selected_features)]
    test_set.head()

    return train_set, test_set

def calculate_inerttias(data):
    inertias = []

    K = range(2, 14)
    for n in K:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        inertias.append(kmeans.inertia_)

    return inertias


def optimal_number_of_clusters(inertias):
    K = range(2, 14)

    y_es =[]
    for i in K:
        y1 = inertias[len(inertias)-1]
        y0 = inertias[0]
        var_y = y1 - y0

        x1 = K[-1]
        x0 = K[0]
        var_x = x1-x0

        pte = (var_y/var_x)
        y = pte*(i-2) + y0
        y_es.append(y)


    distancias = []

    for i in range(len(inertias)-1):
        dist = distance.euclidean(inertias[i],y_es[i])
        distancias.append(dist)

    distancias.append(0)

    for i,value in enumerate(distancias):
        if value == max(distancias):
            optimal = i+2

    return optimal



def create_clusters(data_df, SEED, TEST_SIZE,TARGET_COL):



    X = data_df.iloc[:, 1:]

    inertias = calculate_inerttias(X)

    optimal = optimal_number_of_clusters(inertias)

    n_clusters = optimal + 1
    each_len_set = {0}

    print('\nOPTIMAL NUMBER OF CLUSTERS: ', optimal)

    while (min(each_len_set) < 4):
        n_clusters = n_clusters-1
        print("\nNUMBER OF CLUSTERS: ", n_clusters )


        # define the initial KMEans model
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_jobs=-1,
            random_state=SEED,
        )

        # print('\n[KMEANS]', kmeans)

        target_col_idx = data_df.columns.get_loc(TARGET_COL)
        X = data_df.iloc[:, 1:]

        # Fit Kmeans model
        kmeans = kmeans.fit(X)

        ls_kmeans = list(kmeans.labels_)
        # ls_kmeans_set = set(ls_kmeans)
        # print(ls_kmeans_set)

        each_len = [ls_kmeans.count(x) for x in ls_kmeans]
        each_len_set = set(each_len)

        print("\tSETS: ", each_len_set)

        #alert to show very dissimilar molecules:

        alerts = [j for j,x in enumerate(ls_kmeans) if ls_kmeans.count(x)<4]
        print("\tALERTS!!")
        for alert in alerts:

            print("\t\tcompound number: ", alert, "\n\t\t\tSMILE: ", data_df.iloc[alert]['SMILES'] )

        ################################## checkpoint #########################
        print(alerts)


        if len(alerts)>0:
            print("you have some molecular alerts. It means that these molecules are quite dissimilar")
            print("You can (1) eliminate them or (2) maintain them")


            second_checkpoint = input("What is your choice (1/2)?")

            if second_checkpoint == '1':
                print("Ok,eliminating")
                print(data_df.shape)
                data_df = data_df.drop(data_df.index[alerts], axis=0)
                print(data_df.shape)
                n_clusters = 6
                pass
            else:
                print("Ok, continue with entire dataframe.")
                pass

        #######################################################################






    # add clusters in original dataset in order to split each cluster

    df2 = data_df.assign(cluster = ls_kmeans)
    print(df2)

    cluster_dict = dict()

    for i in range(n_clusters):
        print(i)
        cluster_name = "cluster"+str(i)
        setattr(current_module, cluster_name, df2[df2['cluster']==i])
        cluster_dict[cluster_name] = df2[df2['cluster']==i]


    cluster_train_dict = dict()

    cluster_train_list = []
    cluster_test_dict = dict()
    cluster_test_list = []

    for i, (cluster_name,cluster_value) in enumerate(cluster_dict.items()):
        train_cluster_name = "train_cl"+str(i)
        test_cluster_name = "test_cl"+str(i)
        setattr(current_module, train_cluster_name,pd.DataFrame)
        setattr(current_module, test_cluster_name,pd.DataFrame)

        cluster_train_dict[train_cluster_name],  cluster_test_dict[test_cluster_name] = split_cluster(cluster_value, SEED, TEST_SIZE,TARGET_COL)


        cluster_train_list.append(cluster_train_dict[train_cluster_name])
        cluster_test_list.append(cluster_test_dict[test_cluster_name])



    train_set = pd.concat(cluster_train_list)

    test_set = pd.concat(cluster_test_list)

    print()


    return data_df,train_set, test_set
