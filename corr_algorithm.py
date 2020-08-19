# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/19 10:03
@Auth ： Chyu
@Description：相关性计算方法

"""
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
from data_preprocess import zeroSupplement
from data_preprocess import dataProcessing
from data_preprocess import extractValue
import pandas as pd
import matplotlib.pyplot as plt

#关联度计算
def correlation_algorithm(data):
    corr = {}
    dataAfterProcessing = dataProcessing((data['data']), 25*data['simhash'])
    dataAfterZeroSupp = zeroSupplement(dataAfterProcessing)
    dataValueExtract = extractValue(dataAfterZeroSupp)
    data_proc = np.array(dataValueExtract)
    if data["corr_method"] == "one-to-many":
        arr_list = []
        arr_list2 = []
        arr_list3 = []
        for i in range(len(data_proc)):
            correlation_list = np.corrcoef(data_proc[0],data_proc[i])
            arr_list.append(correlation_list[0][1])
            #不同的相关系数计算方法
            scipy_pea = spearmanr(data_proc[0],data_proc[i])
            scipy_kendalltau = kendalltau(data_proc[0],data_proc[i])
            arr_list2.append(scipy_pea[0])
            arr_list3.append(scipy_kendalltau[0])

        print("arr_list:",arr_list)
        print("arr_list2:", arr_list2)
        print("arr_list3:", arr_list3)
        corr["corr"] = str(arr_list)
    elif data["corr_method"] == "many-for-one":
        # corr_res = pd.DataFrame(data_proc).corr()
        correlation_list = np.corrcoef(data_proc)
        xh_all = []
        for m in range(len(data_proc)):
            xh = []
            for n in range(len(data_proc)):
                xh.append(correlation_list[m][n])
            xh_all.append(xh)
        plt_hotMap(xh_all)
        corr["corr"] = str(xh_all)
    return corr

#热力图
def plt_hotMap(data):
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    # ax.set_yticks(range(len(yLabel)))
    # ax.set_yticklabels(yLabel, fontproperties=font)
    # ax.set_xticks(range(len(xLabel)))
    # ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot

    # hot
    # 从黑平滑过度到红、橙色和黄色的背景色，然后到白色。
    # cool
    # 包含青绿色和品红色的阴影色。从青绿色平滑变化到品红色。
    # gray
    # 返回线性灰度色图。
    # bone
    # 具有较高的蓝色成分的灰度色图。该色图用于对灰度图添加电子的视图。
    # white
    # 全白的单色色图。
    # spring
    # 包含品红和黄的阴影颜色。
    # summer
    # 包含绿和黄的阴影颜色。
    # autumn
    # 从红色平滑变化到橙色，然后到黄色。
    # winter
    # 包含蓝和绿的阴影色。
    im = ax.imshow(data, cmap=plt.cm.gray_r)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title("This is a title")
    # show
    plt.show()
