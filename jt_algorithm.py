# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 10:03
@Auth ： Chyu
@Description：聚类和相关性分析方法入口

"""
from data_preprocess import zeroSupplement
from data_preprocess import dataProcessing
from data_preprocess import extractValue
from cluster_algorithm import clustering_method_choose
from model_evaluation import model_value
from dimen_reduction import getDimAndGetCenter
from corr_algorithm import correlation_algorithm

# import cluser_data from json
def getJson(jsonData):
    res = {}
    try:
        dataAfterProcessing = dataProcessing((jsonData['data']), 10*jsonData['thrhd'])
        dataAfterZeroSupp = zeroSupplement(dataAfterProcessing)
        dataValueExtract = extractValue(dataAfterZeroSupp)
        # choose clusteing method:k-means;DBSCAN;GMM;birch;FCM;SC;AC
        label_pred = clustering_method_choose(jsonData['method'], dataValueExtract, jsonData['K'],
                                              jsonData['parameter'])
        # CorrelationMat = ifShowPearsonMat(jsonData["yourChoose"],dataValueExtract)
        res['label'] = str(list(label_pred)).replace(" ", "")
        if jsonData['original_label'] != -1:
            accuracy,y_pred_p = model_value(jsonData['original_label'], label_pred)
            # label_pred, precision, confusionMatrix, report = model_val(jsonData['method'], jsonData['original_label'], label_pred)
            # res['precision'] = str(precision)
            # res['confusionMatrix'] = str(confusionMatrix).replace("\n","")
            # res['report'] = str(report).replace("\n","")
            res['accuracy'] = str(accuracy)
            res['label'] = str(list(y_pred_p)).replace(" ", "")
        data, draw_centroids = getDimAndGetCenter(dataValueExtract, label_pred)
        print(draw_centroids)
        # res['CorrelationMat'] =str(CorrelationMat).replace(" ", "").replace("\\n"," ")
        res['label'] = str(list(label_pred)).replace(" ", "")
        res['centers'] = str(list(draw_centroids)).replace("  ", ",").replace(" ", "").replace(",,", ",").replace(
            "array(", "").replace(")", "")
        res['data'] = str(list(data)).replace("  ", ",").replace(" ", "").replace(",,", ",").replace("array(",
                                                                                                         "").replace(")",
                                                                                                                     "")

    except Exception as e:
        print(e)
        res['error'] = "Algrithm error :" + str(e)
        pass
    # print(res)
    return res

# import corr_data from json
def correlation(data):
    return correlation_algorithm(data)
