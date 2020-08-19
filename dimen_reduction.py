# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 10:03
@Auth ： Chyu
@Description：降维和返回聚类中心

"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# dimensionality reduction
def dimen_reduction(method, dimension, data):
    if method == 'PCA':
        data_reduc = dimen_reduc_PCA(dimension, data)
    elif method == 'TSNE':
        data_reduc = dimen_reduc_TSNE(dimension, data)
    return data_reduc

# PCA dimensionality reduction
def dimen_reduc_PCA(n, dataSet):
    pca = PCA(n_components=n)
    dataSet = pca.fit_transform(dataSet)
    return dataSet


# TSNE dimensionality reduction
def dimen_reduc_TSNE(n, dataSet):
    tsne = TSNE(n_components=n)
    dataSet = tsne.fit_transform(dataSet)
    return dataSet

# return centers of clusters
def getDimAndGetCenter(data, lableList):
    data = dimen_reduction("PCA", 3, data)
    k = np.max(lableList) + 1
    tmps = [[] for l in range(k)]
    centers = [[] for l in range(k)]

    for j in range(k):
        for i in range(len(lableList)):
            if lableList[i] == j:
                tmps[j].append(data[i])
    np_tmps = np.array(tmps)
    for i in range(k):
        centers[i] = np.mean(np.array(np_tmps[i]), 0)
    # centers = np.array(centers)
    return data, centers[:len(centers)]