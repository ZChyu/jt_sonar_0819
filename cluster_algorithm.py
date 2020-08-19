# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 10:03
@Auth ： Chyu
@Description：包含所有聚类算法

"""
import os
import numpy as np
import math
from scipy.special import gamma
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from dimen_reduction import dimen_reduction

def process_para(parameter):
    for k,v in parameter.items():
        if v == "None":
            print(k,v)
            parameter[k] = None
    print(parameter)
    return parameter

def clustering_method_choose(method, data, k,parameter):
    parameter = process_para(parameter)
    if method == 'k-means':
        label_pred = kmeans(k, data,parameter)
    elif method == 'DBSCAN':
        label_pred = dbscan(data,parameter)
    elif method == 'GMM':
        label_pred = GMM(k, data,parameter)
    elif method == 'BRICH':
        label_pred = brich(k, data,parameter)
    elif method == 'FCM':
        label_pred = FCM(k, data,parameter)
    elif method == 'SC':
        label_pred = SC(k, data,parameter)
    elif method == 'AC':
        label_pred = AC(k, data, parameter)
    return label_pred

# k-means clustering algorithm
def kmeans(k, data, parameter):
    init = parameter["init"]
    n_init = parameter["n_init"]
    max_iter = parameter['max_iter']
    tol = parameter["tol"]
    precompute_distances = parameter["precompute_distances"]
    verbose = parameter["verbose"]
    random_state = parameter["random_state"]
    copy_x = parameter['copy_x']
    n_jobs = parameter["n_jobs"]
    algorithm = parameter["algorithm"]

    km = KMeans(n_clusters=k, init=init, n_init=n_init,
                max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,
                verbose=verbose, random_state=random_state, copy_x=copy_x,
                n_jobs=n_jobs, algorithm=algorithm)

    # km = KMeans(n_clusters=k, init='k-means++', n_init=10,
    #             max_iter=300, tol=1e-4, precompute_distances='auto',
    #             verbose=1, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
    km.fit(data)  # k-means clustering
    labels = km.labels_  # get label
    return labels

# DBSCAN clustering algorithm
def dbscan(data, parameter):
    min_samples = parameter['min_samples']
    metric = parameter['metric']
    metric_params = parameter['metric_params']
    algorithm = parameter['algorithm']
    leaf_size = parameter['leaf_size']
    p = parameter['p']
    n_jobs = parameter['n_jobs']
    grid = int(parameter['gamma'])
    q = parameter['q']

    dataSet = dimen_reduction('PCA', 3, data)
    max_item = max(max(row) for row in dataSet)
    min_item = min(min(row) for row in dataSet)
    data_prod = np.prod(max_item - min_item)

    # eps = math.pow(data_prod*16*gamma(0.5*len(dataSet[1]) + 1)/(len(dataSet)*math.sqrt(math.pow(math.pi, len(dataSet[1])))), 1.1/len(dataSet[1]))
    # db = DBSCAN(eps = eps, min_samples = 3).fit(dataSet)

    eps = math.pow(data_prod * grid * gamma(0.5 * len(dataSet[1]) + 1) / (
    len(dataSet) * math.sqrt(math.pow(math.pi, len(dataSet[1])))), q / len(dataSet[1]))
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p,
                n_jobs=n_jobs).fit(dataSet)

    labels = db.labels_
    return labels


# GaussianMixture clustering algorithm
def GMM(k, data, parameter):
    covariance_type = parameter['covariance_type']
    n_init = parameter['n_init']
    max_iter = parameter['max_iter']
    tol = parameter['tol']
    reg_covar = parameter['reg_covar']
    init_params = parameter['init_params']
    weights_init = parameter['weights_init']
    means_init = parameter['means_init']
    precisions_init = parameter['precisions_init']
    random_state = parameter['random_state']
    warm_start = parameter['warm_start']
    verbose = parameter['verbose']
    verbose_interval = parameter['verbose_interval']

    # gmm = GaussianMixture(n_components=k, covariance_type='full', tol=1e-3,
    #                       reg_covar=1e-6, max_iter=300, n_init=1, init_params='kmeans',
    #                       weights_init=None, means_init=None, precisions_init=None,
    #                       random_state=None, warm_start=False, verbose=0, verbose_interval=10)

    gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, tol=tol,
                          reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                          weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                          random_state=random_state, warm_start=warm_start,
                          verbose=verbose, verbose_interval=verbose_interval)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels


# BIRCH clustering algorithm
def brich(k, data, parameter):
    threshold = parameter['threshold']
    branching_factor = parameter['branching_factor']
    copy = parameter['copy']
    compute_labels = parameter['compute_labels']

    # clf = Birch(threshold=0.5, branching_factor=50,
    #             n_clusters=k, compute_labels=True, copy=True)

    clf = Birch(n_clusters=k, threshold=threshold,
                branching_factor=branching_factor, copy=copy, compute_labels=compute_labels)

    clf.fit(data)
    labels = clf.predict(data)
    return labels


# FCM clustering algorithm
def FCM(k, data, parameter):
    Membership = parameter['Membership']  # default #Membership = 3
    p = parameter['p']
    q = parameter['q']
    max_item = max(max(row) for row in data)
    min_item = min(min(row) for row in data)
    data_prod = np.prod(max_item - min_item)

    # eps = math.pow(data_prod*16*gamma(0.5*len(data[1]) + 1)/(len(data)*math.sqrt(math.pow(math.pi, len(data[1])))), 4/len(data[1]))
    eps = math.pow(
        data_prod * p * gamma(0.5 * len(data[1]) + 1) / (len(data) * math.sqrt(math.pow(math.pi, len(data[1])))),
        q / len(data[1]))
    membership_mat = np.random.random((len(data), int(k)))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])
    while True:
        working_membership_mat = membership_mat ** Membership
        Centroids = np.divide(np.dot(working_membership_mat.T, data),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
        n_c_distance_mat = np.zeros((len(data), k))
        for i, x in enumerate(data):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)
        new_membership_mat = np.zeros((len(data), k))

        for i, x in enumerate(data):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum(
                    (n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (1 / (Membership - 1)))

        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            labels = np.argmax(new_membership_mat, axis=1)
            print("----", new_membership_mat)
            break


    print("--",labels)
    return labels


# StandardSca clustering algorithm
def SC(k, data, parameter):
    eigen_solver = parameter['eigen_solver']
    # n_components = parameter['n_components']
    n_init = parameter['n_init']
    random_state = parameter['random_state']
    gamma = parameter['gamma']
    affinity = parameter['affinity']
    n_neighbors = parameter['n_neighbors']
    eigen_tol = parameter['eigen_tol']
    assign_labels = parameter['assign_labels']
    degree = parameter['degree']
    coef0 = parameter['eigen_tol']
    kernel_params = parameter['kernel_params']
    n_jobs = parameter['n_jobs']

    # SC = SpectralClustering(n_clusters=k, eigen_solver=None, n_components=k-4,
    #                         random_state=1, n_init=10, gamma=0.2, affinity='rbf',
    #                         n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans',
    #                         degree=3, coef0=1, kernel_params=None, n_jobs=None)

    SC = SpectralClustering(n_clusters=k, eigen_solver=eigen_solver,
                            random_state=random_state, n_init=n_init, gamma=gamma, affinity=affinity,
                            n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels,
                            degree=degree, coef0=coef0, kernel_params=kernel_params, n_jobs=n_jobs)

    SC.fit(data)
    labels = SC.fit_predict(data)
    return labels


# StandardSca clustering algorithm
def AC(k, data, parameter):
    affinity = parameter['affinity']
    memory = parameter['memory']
    connectivity = parameter['connectivity']
    compute_full_tree = parameter['compute_full_tree']
    linkage = parameter['linkage']

    # AC = AgglomerativeClustering(n_clusters=k, affinity="euclidean",
    #                               memory=None, connectivity=None,
    #                               compute_full_tree='auto', linkage='ward',
    #                               pooling_func=np.mean)


    AC = AgglomerativeClustering(n_clusters=k, affinity=affinity,
                                 memory=memory, connectivity=connectivity,
                                 compute_full_tree=compute_full_tree, linkage=linkage,
                                 pooling_func=np.mean)

    AC.fit(data)
    labels = AC.fit_predict(data)
    return labels
