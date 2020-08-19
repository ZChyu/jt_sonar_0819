from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from munkres import Munkres
import numpy as np

def model_value(y_true,y_pred):
    y_pred_p = best_map(y_true,y_pred)
    accuracy = accuracy_score(y_true, y_pred_p)
    accuracy_normalize = accuracy_score(y_true, y_pred, normalize=False)
    print(y_true)
    print("预测标签",y_pred)
    print("排序后的预测标签",y_pred_p)
    print("accuracy_score:",accuracy)
    print("accuracy_normalize:", accuracy_normalize)
    print("recall_score",recall_score(y_true, y_pred_p, average='macro'))
    print("precision_score:",precision_score(y_true, y_pred_p, average='micro'))
    print("f1_score:",f1_score(y_true, y_pred_p, average='weighted'))  # F1 score是精确率和召回率的调和平均值
    return accuracy,np.trunc(y_pred_p).astype(int)

def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
	nClass1 = len(Label1)        # 标签的大小
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = int(Label1[c[i]])
	return newL2