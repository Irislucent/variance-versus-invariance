import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_curve


def mscatter_3d(x, y, z, ax=None, m=None, **kw):
    """
    Adapted from https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895
    3d scatter plot with different markers
    """
    import matplotlib.markers as mmarkers

    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, z, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def precision_recall_at_k(emb, labels, k_list=[1, 2, 5, 10, 20, 50, 100, 200, 500]):
    """
    Find the nearest k samples to the input sample
    Compute the precision & recall at k metrics using a list of positive samples
    Also compute the F1 score
    emb: the embeddings of the samples (n_samples, emb_dim)
    labels: the labels of the samples (n_samples,)
    k_list: the list of k values

    Return: the precision, recall, F1 score at k
    """
    n_samples = emb.shape[0]

    all_precisions = []
    all_recalls = []
    all_f1s = []

    for i in range(n_samples):
        # find all the positive samples
        pos_indices = []
        for j in range(n_samples):
            if labels[j] == labels[i] and j != i:
                pos_indices.append(j)

        # compute distances from the input to all the samples
        all_distances = {}
        for j in range(n_samples):
            d = np.linalg.norm(emb[i] - emb[j])
            all_distances[j] = d
        all_distances = sorted(all_distances.items(), key=lambda x: x[1])

        # compute the precision & recall at k metrics
        precision_list = []
        recall_list = []
        f1_list = []
        for k in k_list:
            # find the nearest k samples
            nearest_k = all_distances[:k]
            nearest_k = [x[0] for x in nearest_k]

            # compute the precision and recall at k metric
            precision = 0
            recall = 0
            for j in nearest_k:
                if j in pos_indices:
                    precision += 1
                    recall += 1
            precision /= k
            recall /= len(pos_indices)
            precision_list.append(precision)
            recall_list.append(recall)

            # compute the f1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_list.append(f1)

        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        f1_list = np.array(f1_list)

        all_precisions.append(precision_list)
        all_recalls.append(recall_list)
        all_f1s.append(f1_list)

    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_f1s = np.array(all_f1s)

    all_precisions = np.mean(all_precisions, axis=0)
    all_recalls = np.mean(all_recalls, axis=0)
    all_f1s = np.mean(all_f1s, axis=0)

    return all_precisions, all_recalls, all_f1s


def area_under_prcurve(r_list, p_list):
    """
    Compute the area under the precision-recall curve
    r_list: the list of recall values (should have been sorted in ascending order)
    p_list: the list of precision values (should have been sorted in ascending order)
    """
    area = 0
    for i in range(1, len(r_list)):
        area += (r_list[i] - r_list[i - 1]) * p_list[i]

    return area

