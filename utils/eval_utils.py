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


def precision_recall_at_k(
    emb,
    labels,
    k_list=[1, 2, 5, 10, 20, 50, 100, 200, 500],
    use_gpu=True,
    chunk_size=1024,
    eval_device=None,
):
    """
    Exact precision / recall / f1 at k with chunked GPU computation.

    - preserves exact Euclidean-distance ranking
    - excludes self-match from retrieval
    - avoids materializing a full NxN distance matrix
    """
    emb = torch.as_tensor(emb, dtype=torch.float32)
    labels = torch.as_tensor(labels)

    if eval_device is not None:
        device = torch.device(eval_device)
    else:
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    emb = emb.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    n_samples = emb.shape[0]
    max_k = max(k_list)
    if max_k >= n_samples:
        max_k = n_samples - 1
        k_list = [min(k, max_k) for k in k_list]

    emb_sq = (emb**2).sum(dim=1)

    all_precisions_sum = {k: 0.0 for k in k_list}
    all_recalls_sum = {k: 0.0 for k in k_list}
    all_f1s_sum = {k: 0.0 for k in k_list}

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        bsz = end - start

        emb_chunk = emb[start:end]
        labels_chunk = labels[start:end]
        emb_chunk_sq = (emb_chunk**2).sum(dim=1, keepdim=True)

        dist_chunk = emb_chunk_sq + emb_sq.unsqueeze(0) - 2.0 * (emb_chunk @ emb.T)
        dist_chunk.clamp_(min=0.0)

        row_idx = torch.arange(bsz, device=device)
        col_idx = torch.arange(start, end, device=device)
        dist_chunk[row_idx, col_idx] = float("inf")

        nn_idx_chunk = torch.topk(
            dist_chunk, k=max_k, largest=False, dim=1
        ).indices

        n_pos_chunk = (labels_chunk.unsqueeze(1) == labels.unsqueeze(0)).sum(dim=1) - 1
        n_pos_chunk = n_pos_chunk.clamp(min=1)

        retrieved_labels = labels[nn_idx_chunk]
        retrieved_positive = retrieved_labels == labels_chunk.unsqueeze(1)

        for k in k_list:
            tp = retrieved_positive[:, :k].sum(dim=1).float()
            precision = tp / float(k)
            recall = tp / n_pos_chunk.float()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            all_precisions_sum[k] += precision.sum().item()
            all_recalls_sum[k] += recall.sum().item()
            all_f1s_sum[k] += f1.sum().item()

        del dist_chunk, nn_idx_chunk, retrieved_labels, retrieved_positive, n_pos_chunk

    all_precisions = np.array([all_precisions_sum[k] / n_samples for k in k_list])
    all_recalls = np.array([all_recalls_sum[k] / n_samples for k in k_list])
    all_f1s = np.array([all_f1s_sum[k] / n_samples for k in k_list])

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


def get_confusion_mtx(n_pred, n_labels, preds, labels):
    """
    n_pred: the number of predicted classes (might be more than the true number of classes)
    n_labels: the number of true classes
    preds: the predicted labels
    labels: the true labels
    returns the confusion matrix and the permutation of the predicted classes
    """
    confusion_mtx = np.zeros((n_pred, n_labels))
    for i in range(len(preds)):
        label = int(labels[i])
        pred = int(preds[i])
        confusion_mtx[pred, label] += 1
    confusion_mtx = confusion_mtx / (confusion_mtx.sum(axis=1, keepdims=True) + 1e-7)
    assignments = np.argmax(confusion_mtx, axis=1)
    perm = np.argsort(assignments)
    confusion_mtx = confusion_mtx[perm]

    return confusion_mtx, perm


def confusion_mtx_acc(mtx):
    """
    Overall accuracy of a confusion matrix
    mtx: [n_predicted_classes, n_labels]
    """
    acc = 0
    for i in range(mtx.shape[0]):
        max_entry = mtx[i, :].max()
        acc += max_entry
    acc /= mtx.sum()

    return acc


def confusion_mtx_std(confusion_matrices):
    """
    For a list of confusion matrices (of the same shape, of course), compute the consensus score. They have to have the same permutation.
    horizonally: the real labels
    vertically: the codebook atoms
    """
    if isinstance(confusion_matrices, list):
        confusion_matrices = np.array(confusion_matrices)
    std = np.std(confusion_matrices, axis=0)
    std = np.mean(std)

    return std


def pairwise_d(x):
    """
    x is a tensor of shape (n, d)
    """
    n, d = x.shape
    x1 = x.unsqueeze(0).expand(n, n, d)
    x2 = x.unsqueeze(1).expand(n, n, d)
    stack = torch.stack([x1, x2], dim=0)
    pairwise_d = torch.norm(stack[0] - stack[1], dim=-1)

    return pairwise_d


def denormalize_img(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    x: the normalized data (n_samples, n_channels, height, width), ndarray
    mean: the mean used for normalization
    std: the std used for normalization
    """

    std = np.array(std)
    mean = np.array(mean)
    std = std.reshape((1, -1, 1, 1))
    mean = mean.reshape((1, -1, 1, 1))

    if len(x.shape) == 3:
        x = x.reshape((1,) + x.shape)
        x = x * std + mean
        x = x.reshape(x.shape[1:])
    else:
        x = x * std + mean

    return x


def compute_eer(embeddings, labels, metric="l2"):
    """
    embeddings: the embeddings of the samples
    labels: the labels of the samples
    first compute pairwise distances between embeddings
    then compute the EER
    """
    if metric == "l2":
        d_fn = lambda x, y: np.linalg.norm(x - y)
    else:
        raise NotImplementedError

    scores, answers = [], []
    for i in tqdm(range(len(embeddings))):
        for j in range(i + 1, len(embeddings)):
            scores.append(d_fn(embeddings[i], embeddings[j]))
            answers.append(labels[i] == labels[j])

    scores = np.array(scores)
    scores = 1 - (scores - scores.min()) / (
        scores.max() - scores.min()
    )  # normalize to [0, 1] and invert
    answers = np.array(answers)

    fpr, tpr, thresholds = roc_curve(answers, scores)
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]

    return eer


def mean_and_std(*x):
    x = [np.array(i) for i in x]
    mean = np.mean(x, axis=0) * 100
    std = np.std(x, axis=0) * 100

    # round to 4 decimal places
    mean = np.round(mean, 4)
    std = np.round(std, 4)

    return mean, std
