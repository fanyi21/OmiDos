from typing import Optional
import random
import numpy as np
import pandas as pd
import scipy
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score


# ============== Data Processing ==============
# =============================================

def read_labels(ref, return_enc=False):
    """
    Read labels and encode to 0, 1 .. k with class names 
    """
    # if isinstance(ref, str):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes

def gene_filter_(data, X=6):
    """
    Gene filter in SC3:
        Removes genes/transcripts that are either (1) or (2)
        (1). rare genes/transcripts:
            Expressed (expression value > 2) in less than X% of cells 
        (2). ubiquitous genes/transcripts:
            Expressed (expression value > 0) in at least (100 - X)% of cells
    Input data:
        data is an array of shape (p,n)
    """
    total_cells = data.shape[1]
    count_1 = data[data > 1].count(axis=1)
    count_2 = data[data > 0].count(axis=1)

    genelist_1 = count_1[count_1 > 0.01*X * total_cells].index
    genelist_2 = count_2[count_2 < 0.01*(100-X) * total_cells].index
    genelist = set(genelist_1) & set(genelist_2)
    data = data.loc[genelist]
    return data

def sort_by_mad(data, axis=0):
    """
    Sort genes by mad to select input features
    """
    genes = data.mad(axis=axis).sort_values(ascending=False).index
    if axis==0:
        data = data.loc[:, genes]
    else:
        data = data.loc[genes]
    return data


# =========== scATAC Preprocessing =============
# ==============================================
def peak_filter(data, x=10, n_reads=2):
    count = data[data >= n_reads].count(1)
    index = count[count >= x].index
    data = data.loc[index]
    return data

def cell_filter(data):
    thresh = data.shape[0]/50
    # thresh = min(min_peaks, data.shape[0]/50)
    data = data.loc[:, data.sum(0) > thresh]
    return data

def sample_filter(data, x=10, n_reads=2):
    data = peak_filter(data, x, n_reads)
    data = cell_filter(data)
    return data

# =================== Other ====================
# ==============================================

from scipy.sparse.linalg import eigsh
def estimate_k(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (p,n) matrix, p is feature, n is sample
    """
    p, n = data.shape

    x = scale(data, with_mean=False)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals, _ = eigsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k

# def estimate_k(data):
#     return min(data.shape[0]/100, 30)

def get_decoder_weight(model_file):
    state_dict = torch.load(model_file)
    weight = state_dict['decoder.reconstruction.weight'].data.cpu().numpy()
    return weight

def peak_selection(weight, weight_index, kind='both', cutoff=2.5):
    """
    Select represented peaks of each components of each peaks, 
    correlations between peaks and features are quantified by decoder weight,
    weight is a Gaussian distribution, 
    filter peaks with weights more than cutoff=2.5 standard deviations from the mean.

    Input:
        weight: weight of decoder
        weight_index: generated peak/gene index. 
        kind: both for both side, pos for positive side and neg for negative side.
        cutoff: cutoff of standard deviations from mean.
    """
    std = weight.std(0)
    mean = weight.mean(0)
    specific_peaks = []
    for i in range(10):
        w = weight[:,i]
        if kind == 'both':
            index = np.where(np.abs(w-mean[i]) > cutoff*std[i])[0]
        if kind == 'pos':
            index = np.where(w-mean[i] > cutoff*std[i])[0]
        if kind == 'neg':
            index = np.where(mean[i]-w > cutoff*std[i])[0]
        specific_peaks.append(weight_index[index])
    return specific_peaks
    

def pairwise_pearson(A, B):
    from scipy.stats import pearsonr
    corrs = []
    for i in range(A.shape[0]):
        if A.shape == B.shape:
            corr = pearsonr(A.iloc[i], B.iloc[i])[0]
        else:
            corr = pearsonr(A.iloc[i], B)[0]
        corrs.append(corr)
    return corrs

# ================= Metrics ===================
# =============================================

def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred==i)] = j
        return y_
    from sklearn.utils.linear_assignment_ import linear_assignment
#     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)


def cluster_report(ref, pred, classes=None):
    """
    Print Cluster Report
    """
    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    print('\n## Confusion matrix ##\n')
    print(cm)
    print('\n## Cluster Report ##')
#     print(classification_report(ref, pred, target_names=classes))
    print("Adjusted Rand Index score: {:.4f}".format(adjusted_rand_score(ref, pred)))
    print("`Normalized Mutual Info score: {:.4f}".format(normalized_mutual_info_score(ref, pred)))
    
def binarization(imputed, raw):
    return scipy.sparse.csr_matrix((imputed.T > raw.mean(1).T).T & (imputed>raw.mean(0))).astype(np.int8)

def seed_everything(seed: int, use_cuda: Optional[bool] = True):
    """
    Set the random seed for multiple libraries to ensure reproducible results.

    This function sets the seed for the following libraries:
    - PyTorch (CPU and GPU)
    - NumPy
    - Python's random module
    It also disables PyTorch's CuDNN optimization to further ensure reproducibility.

    :param seed: int, the seed value to be used for all libraries.
    :param use_cuda: bool, optional, default is True. Whether to set seeds for CUDA operations.
    """
    torch.manual_seed(seed)                 # Set seed for current CPU
    np.random.seed(seed)                    # Set seed for Numpy module
    random.seed(seed)                       # Set seed for Python random module

    if use_cuda:
        torch.cuda.manual_seed(seed)        # Set seed for current GPU
        torch.backends.cudnn.benchmark = False  # Disable CuDNN optimization
        torch.backends.cudnn.deterministic = True # Enable deterministic CuDNN behavior
        torch.cuda.manual_seed_all(seed)    # Set seed for all GPUs (optional)
    else:
        torch.use_deterministic_algorithms(True) # Enable deterministic algorithms for CPU operations