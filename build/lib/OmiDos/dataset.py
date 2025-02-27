import pandas as pd
import scanpy as sc
from glob import glob
import os
from anndata import AnnData
import anndata as ad
import scipy
from scipy.sparse import issparse
import muon as mu
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

def LoadSingleSample(path, trans=True):  
    """
    Load single cell dataset from file
    """

    if os.path.exists(path+'.h5ad'):
        adata = sc.read_h5ad(path+'.h5ad')
    elif os.path.isdir(path): # 10X format
        adata = sc.read_10x_mtx(path)
        print("data shape: ", adata.shape)
    elif os.path.isfile(path):
        # if path.endswith(('.csv', '.csv.gz')):
        #     adata = sc.read_csv(path).T
        # elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
        #     df = pd.read_csv(path, sep='\t', index_col=0).T
        #     adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        if path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
    elif path.endswith(tuple(['.h5mu/rna', '.h5mu/atac'])):
        adata = mu.read(path)
    else:
        raise ValueError("File {} not exists".format(path))
    if trans:
        adata = adata.transpose()
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata

def LoadMultiSample(root=None, selected_files=None, file_paths=None):
    """
    Load single cell dataset from files and ensure unique observation names globally by including file info in the names,
    using anndata.concat for concatenation.
    
    :param root: Directory path with or without a wildcard (*) to specify file pattern.
    :param selected_files: Optional list of filenames to specifically include.
    :return: An AnnData object combined from loaded files.
    """
    adata_list = []

    # Handle file paths if directly provided
    if file_paths:
        for file_path in file_paths:
            print("Loading file: " + file_path)
            single_adata = LoadSingleSample(file_path)
            # file_identifier = os.path.basename(file_path).replace('.h5ad', '')
            file_identifier = os.path.basename(os.path.dirname(file_path))
            single_adata.obs_names = [f"{name}-{file_identifier}" for name in single_adata.obs_names]
            single_adata.obs['batch'] = file_identifier
            adata_list.append(single_adata)
            print("===")
            # Concatenate all AnnData objects using anndata.concat
        combined_adata = ad.concat(adata_list, join='outer', label='batch', keys=[adata.obs['batch'].iloc[0] for adata in adata_list])
        return combined_adata
        
    # Handle directory path with potential wildcard and optional selected files
    elif root:
        if root.split('/')[-1] == '*':
            
            file_paths = sorted(glob(root))
            if selected_files is not None:
                file_paths = [path for path in file_paths if path.split('/')[-1] in selected_files]
            
            for file_path in file_paths:
                print("load file: " + file_path)
                single_adata = LoadSingleSample(file_path)
                # Extract the file name without path to use as a unique identifier
                file_identifier = file_path.split('/')[-1].replace('.h5ad', '')
                # Ensure observation names are unique by appending file identifier
                single_adata.obs_names = [f"{name}-{file_identifier}" for name in single_adata.obs_names]
                # Add a 'batch' column to the observations (obs) DataFrame
                single_adata.obs['batch'] = file_identifier
                adata_list.append(single_adata)
                print("===")
            
            # Concatenate all AnnData objects using anndata.concat
            combined_adata = ad.concat(adata_list, join='outer', label='batch', keys=[adata.obs['batch'].iloc[0] for adata in adata_list])
            return combined_adata
        else:
            single_adata = LoadSingleSample(root)
            file_identifier = root.split('/')[-1].replace('.h5ad', '')
            single_adata.obs_names = [f"{name}-{file_identifier}" for name in single_adata.obs_names]
            single_adata.obs['batch'] = file_identifier
            return single_adata
    
def PrepscRNA(adata, 
              min_cells,
              min_genes,
            size_factors=True, 
            logtrans_input=True):

    adata.var_names_make_unique()

    print('Raw RNA dataset shape: {}'.format(adata.shape))

    sc.pp.filter_genes(adata, min_counts=min_cells)
    sc.pp.filter_cells(adata, min_counts=min_genes)
    
    # Saving count data
    adata.layers["raw"] = adata.X.copy()

    # Normalize the data per cell
    if size_factors:
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    # Apply log1p transformation to the data
    if logtrans_input:
        sc.pp.log1p(adata)
    print('Preprocess RNA dataset shape: {}'.format(adata.shape))
    return adata

def ConvertRNATorchDataset(adata, batch_size=128):

    if scipy.sparse.issparse(adata.X) & scipy.sparse.issparse(adata.layers["raw"]):
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X.toarray()), 
                torch.Tensor(adata.layers["raw"].toarray()), 
                torch.Tensor(adata.obs.size_factors)
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    elif scipy.sparse.issparse(adata.X) & ~scipy.sparse.issparse(adata.layers["raw"]):
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X.toarray()), 
                torch.Tensor(adata.layers["raw"]), 
                torch.Tensor(adata.obs.size_factors)
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    elif ~scipy.sparse.issparse(adata.X) & scipy.sparse.issparse(adata.layers["raw"]):
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X), 
                torch.Tensor(adata.layers["raw"].toarray()), 
                torch.Tensor(adata.obs.size_factors)
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    else:
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X), 
                torch.Tensor(adata.layers["raw"]), 
                torch.Tensor(adata.obs.size_factors)
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    # testloader = DataLoader(TensorDataset(torch.Tensor(adata.X.toarray())), batch_size=batch_size, drop_last=False, shuffle=False)

    return trainloader

def ConvertRNATorchBatchDataset(adata, batch_size=128):

    if scipy.sparse.issparse(adata.X) & scipy.sparse.issparse(adata.layers["raw"]):
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X.toarray()), 
                torch.Tensor(adata.layers["raw"].toarray()), 
                torch.Tensor(adata.obs.size_factors),
                torch.Tensor(adata.obsm['onehot'])
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    elif scipy.sparse.issparse(adata.X) & ~scipy.sparse.issparse(adata.layers["raw"]):
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X.toarray()), 
                torch.Tensor(adata.layers["raw"]), 
                torch.Tensor(adata.obs.size_factors),
                torch.Tensor(adata.obsm['onehot'])
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    elif ~scipy.sparse.issparse(adata.X) & scipy.sparse.issparse(adata.layers["raw"]):
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X), 
                torch.Tensor(adata.layers["raw"].toarray()), 
                torch.Tensor(adata.obs.size_factors),
                torch.Tensor(adata.obsm['onehot'])
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    else:
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(adata.X), 
                torch.Tensor(adata.layers["raw"]), 
                torch.Tensor(adata.obs.size_factors),
                torch.Tensor(adata.obsm['onehot'])
            ), 
            batch_size=batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
            )
    # testloader = DataLoader(TensorDataset(torch.Tensor(adata.X.toarray())), batch_size=batch_size, drop_last=False, shuffle=False)

    return trainloader

# Function to convert a scipy sparse matrix to a PyTorch sparse tensor
def scipy_sparse_to_torch_sparse(matrix):

    matrix = matrix.tocoo().astype(np.float32)

    indices = torch.tensor([matrix.row, matrix.col])

    values = torch.tensor(matrix.data)

    shape = torch.Size(matrix.shape)

    return torch.sparse_coo_tensor(indices, values, shape)

def ConvertTorchSparseDataset(adata, batch_size=128):

    X_tensor = scipy_sparse_to_torch_sparse(adata.X)
    
    if 'raw' in adata.layers:
        raw_tensor = scipy_sparse_to_torch_sparse(adata.layers['raw'])
    else:
        raw_tensor = X_tensor  # Use X_tensor if raw data is not separately provided

    size_factors_tensor = torch.tensor(adata.obs['size_factors'].values).float()

    if size_factors_tensor.ndim == 1:
        size_factors_tensor = size_factors_tensor.unsqueeze(1)
    
    train_dataset = TensorDataset(X_tensor, raw_tensor, size_factors_tensor)  
    trainloader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             drop_last=True, 
                             shuffle=True, 
                             num_workers=4)
    
    test_dataset = TensorDataset(X_tensor)  # Placeholder for actual sparse handling
    testloader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            drop_last=False, 
                            shuffle=False)

    return trainloader, testloader

def PrepscATAC(
        adata, 
        min_genes=200, 
        min_cells=3,
        n_top_genes=30000,
        log=None
    ):
    """
    preprocessing
    """
    print('Raw ATAC dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
        
    adata.X[adata.X>1] = 1
    
    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    if log: log.info('Filtering genes')
    if min_cells < 1:
        min_cells = min_cells * adata.shape[0]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if n_top_genes != -1:
        if log: log.info('Finding variable features')
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
        # adata = epi.pp.select_var_feature(adata, nb_features=n_top_genes, show=False, copy=True)

    print('Processed ATAC dataset shape: {}'.format(adata.shape))
    return adata

def ConvertATACTorchDataset(adata, batch_size=128):
    scdata = SingleCellDataset(adata) # Wrap AnnData into Pytorch Dataset
    trainloader = DataLoader(
        scdata, 
        batch_size=batch_size, 
        drop_last=True, 
        shuffle=True, 
        num_workers=4
    )
    testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
    return trainloader, testloader

def ConvertATACTorchBatchDataset(adata, batch_size=128):
    scdata = SingleCellBatchDataset(adata) # Wrap AnnData into Pytorch Dataset
    trainloader = DataLoader(
        scdata, 
        batch_size=batch_size, 
        drop_last=True, 
        shuffle=True, 
        num_workers=4
    )
    testloader = DataLoader(SingleCellDataset(adata), batch_size=batch_size, drop_last=False, shuffle=False)
    return trainloader, testloader

class SingleCellDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, adata):
        self.adata = adata
        self.shape = adata.shape
        
    def __len__(self):
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx):
        if issparse(self.adata.X):

            x = self.adata.X[idx].toarray().squeeze()
        else:
            x = self.adata.X[idx].squeeze()
        # domain_id = self.adata.obs['batch'].cat.codes[idx]
#         return x, domain_id, idx
        return x

class SingleCellBatchDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, adata):
        self.adata = adata
        self.shape = adata.shape
        
    def __len__(self):
        return self.adata.X.shape[0]

    def __getitem__(self, idx):
        if issparse(self.adata.X):
            x = self.adata.X[idx].toarray().squeeze()
        else:
            x = self.adata.X[idx].squeeze()
        # domain_id = self.adata.obs['batch'].cat.codes[idx]
        onehot_label = self.adata.obsm['onehot'][idx] if 'onehot' in self.adata.obsm else None
        return x, onehot_label

