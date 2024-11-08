import numpy as np
import pandas as pd
from GB_generation_with_idx import get_GB


def get_single_attribue(data, radius, l):
    n, m = data.shape
    origin_dixs = np.argsort(data[:, l], axis=0)
    data_temp = data[origin_dixs, :]
    neighbor_set = np.zeros((n, n)) - 1

    for i in range(n):
        s1 = data_temp[i, l]
        j = i
        while j <= i and j >= 0:
            s2 = data_temp[j, l]
            D = ((s1 - s2) ** 2) ** 0.5
            if D <= radius:
                j -= 1
            else:
                break
        a = j + 1
        j = i + 1
        while j < n:
            s2 = data_temp[j, l]
            D = ((s1 - s2) ** 2) ** 0.5
            # D = abs(s1 - s2)
            if D <= radius:
                j += 1
            else:
                break
        b = j - 1
        neighbor_set[origin_dixs[i], np.arange(b - a + 1)] = origin_dixs[a : b + 1]

    neighbor_set_temp = np.unique(neighbor_set, axis=0)
    neighbor_ie_l = 0
    for i in range(neighbor_set_temp.shape[0]):
        cur_m = sum(neighbor_set_temp[i, :] > -1)
        cur_ratio = cur_m / n
        neighbor_ie_l += np.log2(1 / cur_ratio) + 1e-9
    neighbor_ie_l /= n
    return neighbor_set, neighbor_ie_l


def GBNOD(X, sigma):
    n, m = X.shape
    GBs = get_GB(X)
    n_gb = len(GBs)
    
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
    n_gb = centers.shape[0]
    ie_ks = np.zeros(m)
    weight = np.zeros((n_gb, m))
    
    for k in range(m):
        radius = np.std(centers[:, k]) / sigma
        neighbor_set, neighbor_ie_k = get_single_attribue(centers, radius, k)    
        weight_x = np.zeros(n_gb)
        for i in range(n_gb):
            weight_x[i] = (sum(neighbor_set[i, :] > -1) / n_gb)
        weight[:, k] = weight_x
        ie_ks[k] = neighbor_ie_k
    gbof = np.zeros(n_gb)
    od = ie_ks / np.sum(ie_ks, axis=0)
    for i in range(n_gb):
        gbof[i] = np.sum(((1 / (1 + weight[i])) * np.cbrt(od))) / m
    OF = np.zeros(n)
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OF[point_idxs] = gbof[idx]
    return OF


if __name__ == '__main__':
    data = pd.read_csv("./Example.csv").values
    sigma = 0.6
    n, m = data.shape
    OS = GBNOD(data, sigma)
    print(OS)
    
#     ie_ks = np.zeros(m)
#     x_ie_ks = np.zeros((n_gb, m))
#     rnc = np.zeros((n_gb, m))
#     weight = np.zeros((n_gb, m))
    
#     # ie_ks = np.zeros(m)
#     # x_ie_ks = np.zeros((n, m))
#     # rnc = np.zeros((n, m))
#     # weight = np.zeros((n, m))

#     best_auc = 0
#     for sigma in np.arange(0.1, 2, 0.05):
#         for k in range(m):
#             radius = np.std(centers[:, k]) / sigma
#             # radius=2
#             # radius = 0.5
#             # print("radius", radius)
#             # radius = sigma
#             neighbor_set, neighbor_ie_k, neighbor_x_ies_k, rnc_x_k = get_single_attribue(centers, radius, k)    
#             weight_x = np.zeros(n_gb)
#             for i in range(n_gb):
#                 # weight_x[i] = np.sqrt(np.count_nonzero(neighbor_set[i, :]) / n)
#                 weight_x[i] = (sum(neighbor_set[i, :] > -1) / n_gb)

#             weight[:, k] = weight_x
#             # rnc[:, k] = rnc_x_k
#             # # NE
#             ie_ks[k] = neighbor_ie_k
#             # # NEx
#             # x_ie_ks[:, k] = neighbor_x_ies_k
#         # nod = np.zeros((n_gb, m))
#         # for i in range(n_gb):
#         #     rne_x = 1 - x_ie_ks[i] / ie_ks
#         #     rne_x[rne_x < 0] = 0
#         #     rne_x[rne_x > 1] = 0    
#         #     for k in range(m):
#         #         if rnc[i, k] > 0:
#         #             nod[i, k] = rne_x[k] * (n_gb - np.abs(rnc[i, k])) / (2 * n_gb)
#         #         else:
#         #             nod[i, k] = rne_x[k] * np.sqrt((n_gb + np.abs(rnc[i, k])) / (2 * n_gb))
#         gbof = np.zeros(n_gb)
#         od = ie_ks / np.sum(ie_ks, axis=0)
#         for i in range(n_gb):
#             # gbof[i] = 1 - (np.sum((1 - nod[i]) * weight[i])) / (2 * m)
#             # gbof[i] = 1 - np.sum(weight[i]) / m
#             # gbof[i] = 1 - np.sum(np.cbrt((weight[i]) * (1 - od))) / m
#             gbof[i] = np.sum(((1 - weight[i]) * (od))) / m
            
            
        
#         # 映射回样本
#         OF = np.zeros(n)
#         for idx, gb in enumerate(GBs):
#             point_idxs = gb[:,-1].astype('int')
#             OF[point_idxs] = gbof[idx]
#         # print(OF)
#         cur_auc = roc_auc_score(label, OF)
#         print(cur_auc)
#         if cur_auc > best_auc:
#             best_auc = cur_auc
#     print("Best auc:", best_auc)