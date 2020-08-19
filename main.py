# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 10:03
@Auth ： Chyu
@Description：读取数据及调用相关算法

"""
import  xlrd
import os
import matplotlib.pyplot as plt
import pyhttp
import json
import random,copy
import sys
sys.path.append("E://建投算法//sonar_para")
sys.path

pyserviceUrl='http://localhost:5000/'
# serviceurl='jt_pyservice' #jt_kmeans,jt_Agglomerative,,jt_dbscan,jt_GMM,jt_brich,jt_mean_shfit,jt_Affinity_propa,jt_StandardSca
serviceurl='jt_correlation'
url=pyserviceUrl+serviceurl

def read_xlrd(path):
    dataf = []
    datap = []
    data_all = []
    dirs = os.listdir(path)
    for file in dirs:
        dataFile = []
        if os.path.splitext(file)[1] == ".xlsx":
            data = xlrd.open_workbook(path+"//"+file)
            table = data.sheet_by_name("LOFAR特征")
            for rowNum in range(table.nrows):
                dataRow = table.row_values(rowNum)
                dataFile.append(dataRow)

            for colNum in range(table.ncols):
                if colNum == 0:
                    dataFc = table.col_values(colNum)
                if colNum == 1:
                    dataPc = table.col_values(colNum)
            datap.append(dataPc) # 频率
            dataf.append(dataFc) # 幅值
        data_all.append(dataFile)
    return dataf,datap,data_all
if __name__ == '__main__':
    execFile = "D:/workspace/PycharmProjects/sonar_para/data/仿真excel数据/1-5"
    dataf,datap,data_all = read_xlrd(execFile)
    # print(data_all)
    # drawCircle(dataf,datap)
    original_label= []
    for i in range(1,6):
        for j in range(30):
            original_label.append(i)
    # print(original_label)
    #k-means
    parameter = {"init": 'k-means++', "n_init": 10,
                  "max_iter": 300, "tol": 1e-4, "precompute_distances": 'auto',
                  "verbose": 0, "random_state": None, "copy_x": True,
                  "n_jobs": None, "algorithm": 'auto'}
    # FCM
    # parameter ={"Membership":3,"p":16,"q":1.1}
    # GMM parameter
    # parameter = {"covariance_type": 'full', "n_init": 10,
    #              "max_iter": 300, "tol": 1e-3, "reg_covar": 1e-6,
    #              "init_params": 'kmeans', "weights_init": None, "means_init": None,
    #              "precisions_init": None, "random_state": None, "warm_start": False,
    #              "verbose": 0, "verbose_interval": 10}
    # BRICH
    # parameter = {"threshold": 0.5, "branching_factor": 50,
    #               "compute_labels": True, "copy": True}
    # SC
    # parameter =  {"eigen_solver":None, "n_components":None,
    #                  "random_state":None, "n_init":10, "gamma":1., "affinity":'rbf',
    #                  "n_neighbors":10, "eigen_tol":0.0, "assign_labels":'kmeans',
    #                  "degree":3, "coef0":1, "kernel_params":None, "n_jobs":None}
    # DBSCAN
    # parameter = {"min_samples": 5, "q": 1.1, "gamma": 16, "metric": 'euclidean', "metric_params": None,
    #               "algorithm": 'auto', "leaf_size": 30, "p": None, "n_jobs": None}
    # AC
    # parameter = {"affinity":"euclidean",
    #                  "memory":None,
    #                  "connectivity":None, "compute_full_tree":'auto',
    #                  "linkage":'ward'}
    parameters = {'data': data_all, 'simhash': 1.2,'corr_method':"many-for-one"} #many-for-one #one-to-many
    # parameters = {'data': data_all, 'thrhd': 4, 'K': 5, 'method': 'k-means', "parameter": parameter, "original_label": -1}
    # parameters = {'data': ll, 'thrhd': 10, 'K': 7, 'method': 'k-means',"parameter":parameter,"original_label":[ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # import jt_algrith
    # import jt_algorithm
    # res = jt_algorithm.correlation(parameters)
    # res = jt_algorithm.getJson(parameters)
    res = pyhttp.getmodel(url, parameters)

    print(res)