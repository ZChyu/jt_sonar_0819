# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 10:03
@Auth ： Chyu
@Description：数据预处理

"""
import copy
import numpy as np

# data processing: getmode
def getMode(data_point, threshold):
    data_frequency = np.array(data_point[0])
    data_point[0] = data_frequency // threshold * threshold
    return data_point


# data processing: sorted
def dataSorted(data_sample):
    data_sample = sorted(data_sample, key=(lambda x: x[0]))
    return data_sample


# data processing: delete duplication itme
def dataDeleteDuplication(data_sample):
    data_arr = np.array(data_sample)
    data_sample = np.array(list(set([tuple(t) for t in data_arr])))
    return data_sample

# data lofar processing
def dataProcessing(data, threshold):
    for i in range(len(data)):
        data[i] = dataDeleteDuplication(data[i])
        data[i] = dataSorted(data[i])
        for j in range(len(data[i])):
            data[i][j] = getMode(data[i][j], threshold)
    return data


# make-up lofar data with zero
def zeroSupplement(data):
    frequency_set = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][0] not in frequency_set:
                frequency_set.append(data[i][j][0])
    frequency_set = sorted(set(frequency_set))
    data_zero_lofar = []

    for k in range(len(frequency_set)):
        data_zero_lofar.append([frequency_set[k], 0])

    for m in range(len(data)):
        temp = copy.deepcopy(data_zero_lofar)
        for n in range(len(data[m])):
            lofar_index = frequency_set.index(data[m][n][0])
            temp[lofar_index] = data[m][n]
        data[m] = temp
    return data


# extract value from lofar data(including frequency an value)
def extractValue(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j][1]
    return data