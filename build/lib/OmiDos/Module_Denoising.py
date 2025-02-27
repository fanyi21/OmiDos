import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import scanpy as sc


class FilterModule(nn.Module):
    def __init__(self, original_dim, im_dim):
        super(FilterModule, self).__init__()
        self.encoder = nn.Linear(original_dim, im_dim)
        self.decoder = nn.Linear(im_dim, original_dim)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x):
        # x = self.encoder(x)
        x = F.relu(self.encoder(x))
        x = self.decoder(x)
        return x

def RunDenoising(adata, TF, DenosingModule_Dir, original_dim, im_dim=32, num_epochs = 50, filter_num = 5000, batch_size = 256, device = "cuda:0", imputation=False, sample_time=1):
    # , data_type=str()
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     model = nn.DataParallel(model)
    sample_time = sample_time
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    # if data_type == "scRNA":
    inx, db = process_data(adata.X, sparse=TF)
    adata = sc.AnnData(db, obs=adata[:,inx].obs, var=adata[:,inx].var)
    # elif data_type == "scATAC":
    #     inx = None
    #     pass
    # else:
    #     raise ValueError("data_type should be scRNA or scATAC")
    WSD = np.zeros((sample_time, adata.n_vars))
    full_outputs = []
    # three times sampling
    for i in tqdm(range(sample_time)):

        ## doing sample
        # np.random.seed(i+1)
        # if adata.X.shape[0] > 50000: 
        #     adata_sample = adata[np.random.choice(adata.X.shape[0], 5000, replace=False), :]
        #     wdecay = 1e-4
        # elif adata.X.shape[0] > 2000:
        #     adata_sample = adata[np.random.choice(adata.X.shape[0], 2000, replace=False), :]
        #     wdecay = 1e-6
        # else:
        #     adata_sample = adata
        #     wdecay = 1e-6

        ## no sample
        # adata_sample = adata
        wdecay = 1e-6


        model = FilterModule(original_dim=adata.n_vars, im_dim=im_dim)
        model.to(device)
        model.train()
        # batch_size = max(int(np.ceil(adata_sample.X.shape[0] / 50)), 2)
        dataset = TensorDataset(torch.Tensor(adata.X.toarray()))
        # dataset = TensorDataset(torch.from_scipy(adata_sample.X)) # 转换为 PyTorch 稀疏张量
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), 
                                lr=1e-3, 
                                betas=(0.9, 0.999), 
                                eps=1e-7, 
                                weight_decay=wdecay)

        # Train the autoencoder
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            loss_val = 0
            for batch_data in dataloader:
                inputs = batch_data[0].to(device)
                # Forward pass
                outputs = model(inputs)
                # Calculate loss
                loss = criterion(outputs, inputs)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if isinstance(model, nn.DataParallel):
                        model.module.encoder.weight.copy_(model.module.encoder.weight.data.clamp(min=0))
                    else:
                        model.encoder.weight.copy_(model.encoder.weight.data.clamp(min=0))
                loss_val += loss.item() / len(batch_data[0])
            # Print the progress
            if (epoch+1)%100:
                pass
            else:
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_val/adata_sample.X.shape[0]:.4f}')
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_val:.8f}')

        # Save the model
        torch.save(model.state_dict(), DenosingModule_Dir+"/Denoising_"+str(i)+".pt")

        if imputation:
        # Switch to evaluation mode for inference
            model.eval()

            # Collect outputs for inference
            with torch.no_grad():
                
                outputs = []
                full_dataset = TensorDataset(torch.Tensor(adata.X.toarray()))
                full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

                for batch_data in full_dataloader:
                    inputs = batch_data[0].to(device)
                    outputs.append(model(inputs).cpu().numpy())

                # Concatenate all batch outputs
                full_output = np.concatenate(outputs, axis=0)    
                full_outputs.append(full_output)

        # Get the weights of the first linear layer (encoder) and transpose it
        if isinstance(model, nn.DataParallel):
            W = model.module.encoder.weight.detach().cpu().numpy().T
        else:
            W = model.encoder.weight.detach().cpu().numpy().T
        # Compute the row standard deviations
        Wsd = np.std(W, axis=1)
        # Replace NaN values with 0
        Wsd = np.nan_to_num(Wsd)
        # Normalize the standard deviations to the range [0, 1]
        Wsd = (Wsd - np.min(Wsd)) / (np.max(Wsd) - np.min(Wsd))
        WSD[i,:] = Wsd
    Wsd = np.mean(WSD, axis=0)

    if imputation:
        imputation_data = np.mean(full_outputs, axis=0)
        imputation_data = sc.AnnData(imputation_data, obs=adata.obs, var=adata.var)
    else:
        imputation_data = full_outputs


    # Calculate the quantile value
    quantile_val = np.quantile(Wsd, 1 - min(filter_num, original_dim) / original_dim)
    # Get the indices of the elements that are greater than the quantile value
    keep = np.where(Wsd > quantile_val)[0]
    return inx, keep, imputation_data

def normalize_data_dense(data):
    col_sum = (data != 0).sum(axis=0)
    idx = np.where(col_sum > 0)[0]
    data = data[:, idx]
    
    tmp_min = data.min(axis=0)
    tmp_max = data.max(axis=0)
    tmp_sub = tmp_max - tmp_min
    
    data = (data - tmp_min) / tmp_sub
    return idx, csr_matrix(data)

def normalize_data_sparse(data):
    col_sum = (data != 0).sum(axis=0)
    idx = np.where(col_sum == 0)[0]
    data = data[:, np.setdiff1d(np.arange(data.shape[1]), idx)]
    
    tmp_min = data.min(axis=0).A.flatten()
    tmp_max = data.max(axis=0).A.flatten()
    tmp_sub = tmp_max - tmp_min
    
    data = data.toarray()  # convert to dense for operation
    data = (data - tmp_min) / tmp_sub
    return idx,csr_matrix(data)
 
def process_data(data, sparse=False):
    if sparse:
        if np.min(data) < 0:
            raise ValueError("The input must be a non-negative sparse matrix.")
        if np.max(data) - np.min(data) > 100:
            da = np.log2(data.toarray() + 1)
        da = normalize_data_sparse(da)
    else:
        if np.max(data) - np.min(data) > 100:
            if np.min(data) < 0:
                if data.shape[0] == data.shape[1]:
                    data = data.T
                    data = data - np.min(data, axis=1)[:, np.newaxis]
                    data = data.T
                else:
                    data = data - np.min(data, axis=0)
            da = np.log2(data.toarray() + 1)
        idx, da = normalize_data_dense(da)
    return idx, da



