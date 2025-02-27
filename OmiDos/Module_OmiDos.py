import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from tqdm import tqdm, trange
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
from sympy.abc import x

# Define custom loss function
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        loss = torch.mean(torch.where(torch.le(x, 1e-8), zero_case, nb_case))
        return loss

# Activation functions 
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var

def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)

def elbo_ATAC(recon_x, x, gamma, c_params, z_params, binary=True):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    mu_c, var_c, pi = c_params #print(mu_c.size(), var_c.size(), pi.size())
    var_c += 1e-8
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)

    # log p(x|z)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x) #;print(logvar_expand.size()) #, torch.exp(logvar_expand)/var_c)
    else:
        likelihood = -F.mse_loss(recon_x, x)

    # log p(z|c)
    logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           torch.exp(logvar_expand)/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
    
    # log p(c)
    logpc = torch.sum(gamma*torch.log(pi), 1)

    # log q(z|x) or q entropy    
    qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma*torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx
    
    return torch.sum(likelihood), torch.sum(kld)

def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class ShareEncoder(nn.Module):
    def __init__(self, dims, cell_type, bn=False, dropout=0):
        super(ShareEncoder, self).__init__()

        if cell_type == "scRNA":
            [x_dim, y_dim, z_dim, encode_dim, decode_x_dim, decode_y_dim] = dims
        elif cell_type == "scATAC":
            [y_dim, x_dim, z_dim, encode_dim, decode_x_dim, decode_y_dim] = dims
        elif cell_type == "scRNA_scATAC":
            [x_dim, y_dim, z_dim, encode_dim, decode_x_dim, decode_y_dim] = dims
        self.bn = bn
        self.dropout = dropout
        self.x_dim = x_dim
        self.encode_dim = encode_dim
        self.cell_type = cell_type

    def CreateModule(self):
        if self.cell_type == "scRNA" or self.cell_type == "scATAC": 
            shareEncoder = build_mlp([self.x_dim]+self.encode_dim, bn=self.bn, dropout=self.dropout)
        elif self.cell_type == "scRNA_scATAC":
            shareEncoder = build_mlp(self.encode_dim, bn=self.bn, dropout=self.dropout)
        else:
            raise NotImplementedError
        return shareEncoder

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class OmiDosModel(nn.Module):
    def __init__(self, dims, n_centroids, device, binary, cell_type, sigma=1.0, bn=False, dropout=0, output_activation=nn.Sigmoid()):
        super(OmiDosModel, self).__init__()

        [x_dim, y_dim, z_dim, encode_dim, decode_x_dim, decode_y_dim] = dims

        self.device = device
        self.sigma = sigma
        self.bn = bn
        self.dropout = dropout
        self.x_dim = x_dim
        self.encode_dim = encode_dim
        self.decode_x_dim = decode_x_dim
        self.decode_y_dim = decode_y_dim
        self.binary = binary

        self.n_centroids = n_centroids

        self.cell_type = cell_type

        self.Diff = DiffLoss()

        # share encoder
        self.shareEncoder = ShareEncoder(dims, self.cell_type, bn=self.bn, dropout=self.dropout).CreateModule()

        if self.cell_type == "scRNA":

            # # RNA encoder part
            # self._enc_mu = nn.Linear(encode_dim[-1], z_dim)
            # # RNA decoder part
            # self.decoder_RNA = build_mlp([z_dim]+decode_x_dim, bn=self.bn, dropout=self.dropout)
            # # RNA Loss
            # self._dec_mean = nn.Sequential(nn.Linear(decode_x_dim[-1], x_dim), MeanAct())
            # self._dec_disp = nn.Sequential(nn.Linear(decode_x_dim[-1], x_dim), DispAct())
            # self._dec_pi = nn.Sequential(nn.Linear(decode_x_dim[-1], x_dim), nn.Sigmoid())
            # self.zinb_loss = ZINBLoss().to(self.device)
            pass

        elif self.cell_type == "scATAC":
            # init c_params
            # self.pi = nn.Parameter(torch.ones(self.n_centroids)/self.n_centroids)  # pc
            # self.mu_c = nn.Parameter(torch.zeros(z_dim, self.n_centroids)) # mu
            # self.var_c = nn.Parameter(torch.ones(z_dim, self.n_centroids)) # sigma^2

            # # ATAC encoder part
            # self.sample = GaussianSample(([y_dim]+encode_dim)[-1], z_dim)
            # # ATAC decoder part
            # self.decoder_ATAC_hiden = build_mlp([z_dim, *decode_y_dim], bn=self.bn, dropout=self.dropout)
            # self.reconstruction = nn.Linear([z_dim, *decode_y_dim][-1], y_dim)
            # self.output_activation = output_activation

            # self.reset_parameters()
            pass
        elif self.cell_type == "scRNA_scATAC":

            # init c_params
            self.pi = nn.Parameter(torch.ones(self.n_centroids)/self.n_centroids)  # pc
            self.mu_c = nn.Parameter(torch.zeros(z_dim, self.n_centroids)) # mu
            self.var_c = nn.Parameter(torch.ones(z_dim, self.n_centroids)) # sigma^2
            # RNA Loss
            self._dec_mean = nn.Sequential(nn.Linear(decode_x_dim[-1], x_dim), MeanAct())
            self._dec_disp = nn.Sequential(nn.Linear(decode_x_dim[-1], x_dim), DispAct())
            self._dec_pi = nn.Sequential(nn.Linear(decode_x_dim[-1], x_dim), nn.Sigmoid())
            self.zinb_loss = ZINBLoss().to(self.device)
            ######## private RNA Encoder  ###########
            self.private_RNA_encoder = build_mlp([x_dim]+encode_dim, bn=self.bn, dropout=self.dropout)
            self.privat_enc_mu = nn.Linear(encode_dim[-1], z_dim)
            #+++++++++++++++++++++++++++++++++++++++#

            ######## private ATAC Encoder  ##########
            self.private_ATAC_encoder = build_mlp([y_dim]+encode_dim, bn=self.bn, dropout=self.dropout)
            self.private_ATAC_sample = GaussianSample(([y_dim]+encode_dim)[-1], z_dim)
            #+++++++++++++++++++++++++++++++++++++++#

            ########     share Encoder     ##########
            self.layer_RNA = build_mlp([x_dim]+[encode_dim[0]], bn=self.bn, dropout=self.dropout)
            # RNA encoder part
            self._enc_mu = nn.Linear(encode_dim[-1], z_dim)

            # keep the same dim layer ATAC
            self.layer_ATAC = build_mlp([y_dim]+[encode_dim[0]], bn=self.bn, dropout=self.dropout)
            # ATAC encoder part
            self.sample = GaussianSample(([y_dim]+encode_dim)[-1], z_dim)
            #+++++++++++++++++++++++++++++++++++++++#

            ########     share Decoder     ##########  
            # RNA decoder part
            self.decoder_RNA = build_mlp([z_dim]+decode_x_dim, bn=self.bn, dropout=self.dropout)

            # ATAC decoder part
            self.decoder_ATAC_hiden = build_mlp([z_dim, *decode_y_dim], bn=self.bn, dropout=self.dropout)
            self.reconstruction = nn.Linear([z_dim, *decode_y_dim][-1], y_dim)
            self.output_activation = output_activation
            self.reset_parameters()
            #+++++++++++++++++++++++++++++++++++++++#

            ########  classify two domain  ##########
            self.loss_similarity = torch.nn.CrossEntropyLoss()
            self.shared_encoder_pred_domain = nn.Sequential()
            self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=z_dim, out_features=z_dim))
            self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))
            # classify two domain
            self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=z_dim, out_features=2))

        else:
            raise ValueError("Cell type not supported")

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y, mode):
        if self.cell_type not in ["scRNA", "scATAC", "scRNA_scATAC"]:
            raise ValueError("Invalid cell_type")
        if mode not in ["source", "target"]:
            raise ValueError("Invalid mode")

        if self.cell_type == "scRNA":  
            h_x = self.shareEncoder(x + torch.randn_like(x) * self.sigma)
            # RNA Encoder part
            z_x = self._enc_mu(h_x)
            # RNA Decoder part
            # self.recon_x = self.decoder_RNA(z_x)
            recon_x = self.decoder_RNA(z_x)
            return recon_x
        elif self.cell_type == "scATAC":
            # ATAC Encoder part

            # y_0 = self.layer(y)
            # h_y = self.shareEncoder(y_0)

            h_y = self.shareEncoder(y)

            z_y,_,_ = self.sample(h_y)
            # ATAC Decoder part
            z_y = self.decoder_ATAC_hiden(z_y)
            if self.output_activation is not None:
                recon_y = self.output_activation(self.reconstruction(z_y))
            else:
                recon_y = self.reconstruction(z_y)
            return recon_y
        elif self.cell_type == "scRNA_scATAC":
            result = []
            if mode == 'source':
                ######## private RNA Encoder  ###########
                p_x = self.private_RNA_encoder(x + torch.randn_like(x) * self.sigma)
                z_p = self.privat_enc_mu(p_x)
                #+++++++++++++++++++++++++++++++++++++++#
            elif mode == 'target':
                ######## private ATAC Encoder  ##########
                p_y = self.private_ATAC_encoder(y)
                z_p,_,_ = self.private_ATAC_sample(p_y)
                #+++++++++++++++++++++++++++++++++++++++#
            result.append(z_p)
            
            ########     share Encoder     ##########
            if mode == 'source':
                h_x = self.layer_RNA(x + torch.randn_like(x) * self.sigma)
                h_x = self.shareEncoder(h_x + torch.randn_like(h_x) * self.sigma)
                z_s = self._enc_mu(h_x)
            elif mode == 'target':
                h_y = self.layer_ATAC(y)
                h_y = self.shareEncoder(h_y)
                z_s,_,_ = self.sample(h_y)
            #+++++++++++++++++++++++++++++++++++++++#
            result.append(z_s)

            ########  classify two domain  ##########
            domain_label = self.shared_encoder_pred_domain(z_s)
            #+++++++++++++++++++++++++++++++++++++++#
            result.append(domain_label)


            ########     share Decoder     ##########
            recon_x = self.decoder_RNA(z_x)
            result.append(recon_x)

            # ATAC Decoder part
            z_y = self.decoder_ATAC_hiden(z_y)
            if self.output_activation is not None:
                recon_y = self.output_activation(self.reconstruction(z_y))
            else:
                recon_y = self.reconstruction(z_y)
            result.append(recon_y)
            #+++++++++++++++++++++++++++++++++++++++#

            return result
        
    def LossscRNA(self, x, x_raw, size_factor):

        h = self.private_RNA_encoder(x + torch.randn_like(x) * self.sigma)
        z = self.privat_enc_mu(h)
        recon_x = self.decoder_RNA(z)

        _mean = self._dec_mean(recon_x)
        _disp = self._dec_disp(recon_x)
        _pi = self._dec_pi(recon_x)

        loss = self.zinb_loss(x_raw, _mean, _disp, _pi, size_factor)

        return loss

    def LossscATAC(self, y):
        h = self.private_ATAC_encoder(y)
        z, mu, logvar = self.private_ATAC_sample(h)

        z_y = self.decoder_ATAC_hiden(z)
        if self.output_activation is not None:
            recon_y = self.output_activation(self.reconstruction(z_y))
        else:
            recon_y = self.reconstruction(z_y)
        gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
        likelihood, kl_loss = elbo_ATAC(recon_y, y, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

        return -likelihood, kl_loss
    
    def LossShare(self, x, x_raw, size_factor, y, mode):
        if mode == 'source':

            # h = self.private_RNA_encoder(x + torch.randn_like(x) * self.sigma)
            # z_p = self.privat_enc_mu(h)

            h = self.layer_RNA(x + torch.randn_like(x) * self.sigma)
            h = self.shareEncoder(h + torch.randn_like(h) * self.sigma)
            z_s = self._enc_mu(h)

            z = z_s
            # z = z_p + z_s # union

            recon_x = self.decoder_RNA(z)
            _mean = self._dec_mean(recon_x)
            _disp = self._dec_disp(recon_x)
            _pi = self._dec_pi(recon_x)

            loss = self.zinb_loss(x_raw, _mean, _disp, _pi, size_factor)
            return loss
        
        elif mode == 'target':

            # h = self.private_ATAC_encoder(y)
            # z_p, mu, logvar = self.private_ATAC_sample(h)

            h = self.layer_ATAC(y)
            h = self.shareEncoder(h)
            z_s, mu, logvar = self.sample(h)

            z = z_s

            # z = z_p +z_s # union

            z_y = self.decoder_ATAC_hiden(z)
            if self.output_activation is not None:
                recon_y = self.output_activation(self.reconstruction(z_y))
            else:
                recon_y = self.reconstruction(z_y)
            gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
            likelihood, kl_loss = elbo_ATAC(recon_y, y, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

            return -likelihood, kl_loss

    def LossDiff(self, x, y, mode):
        
        if mode == 'source':
            h = self.private_RNA_encoder(x + torch.randn_like(x) * self.sigma)
            z_p = self.privat_enc_mu(h)

            h = self.layer_RNA(x + torch.randn_like(x) * self.sigma)
            h = self.shareEncoder(h + torch.randn_like(h) * self.sigma)
            z_s = self._enc_mu(h)
        elif mode == 'target':
            h = self.private_ATAC_encoder(y)
            z_p,_,_ = self.private_ATAC_sample(h)

            h = self.layer_ATAC(y)
            h = self.shareEncoder(h)
            z_s,_,_ = self.sample(h)

        D_loss = self.Diff(z_p, z_s)
        return D_loss

    def LossDomainLabel(self, x, y, label_x, label_y, mode):
        ########     share Encoder     ##########
        if mode == 'source':
            h_x = self.layer_RNA(x + torch.randn_like(x) * self.sigma)
            h_x = self.shareEncoder(h_x + torch.randn_like(h_x) * self.sigma)
            z_s = self._enc_mu(h_x)
            label = label_x
        elif mode == 'target':
            h_y = self.layer_ATAC(y)
            h_y = self.shareEncoder(h_y)
            z_s,_,_ = self.sample(h_y)
            label = label_y
        #+++++++++++++++++++++++++++++++++++++++#

        ########  classify two domain  ##########
        domain_label = self.shared_encoder_pred_domain(z_s)
        dl_loss = self.loss_similarity(domain_label, label.float())
        return dl_loss

    def EmbedingSharescRNA(self, X, batch_size=256,  out='z'):
        
        self.eval().to(self.device)
        X = torch.Tensor(X)
        encoded = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))

        with torch.no_grad():  # No gradients needed for embedding generation
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                inputs = Variable(xbatch).to(self.device)
                
                # h = self.encoder(inputs + torch.randn_like(inputs) * self.sigma)

                h_x = self.layer_RNA(inputs)
                h = self.shareEncoder(h_x)
                
                z = self._enc_mu(h)
                if out == 'z':
                    encoded.append(z.detach())
                elif out == 'h':
                    encoded.append(h.detach())

        return torch.cat(encoded, dim=0)
    
    def EmbedingSinglescRNA(self, X, batch_size=256):
        
        self.eval().to(self.device)
        X = torch.Tensor(X)
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))

        with torch.no_grad():  # No gradients needed for embedding generation
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                inputs = Variable(xbatch).to(self.device)          
                # h = self.encoder(inputs + torch.randn_like(inputs) * self.sigma)
                h = self.private_RNA_encoder(inputs)
                z = self.privat_enc_mu(h)
                encoded.append(z.detach())

        return torch.cat(encoded, dim=0)

    def EmbedingSinglescATAC(self, dataloader, device='cpu', out='z', transforms=None):
        self.eval().to(self.device)
        output = []
        for x in dataloader:
            x = x.view(x.size(0), -1).float().to(device)

            h = self.private_ATAC_encoder(x)
        
            z, mu, logvar = self.private_ATAC_sample(h)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                # recon_x = self.decoder(z)
                z = self.decoder_ATAC_hiden(z)
                if self.output_activation is not None:
                    recon_x = self.output_activation(self.reconstruction(z))
                else:
                    recon_x = self.reconstruction(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)

        output = torch.cat(output).numpy()

        return output
    
    def EmbedingSharescATAC(self, dataloader, device='cpu', out='z', transforms=None):
        self.eval().to(self.device)
        output = []
        for x in dataloader:
            x = x.view(x.size(0), -1).float().to(device)
            # z, mu, logvar = self.encoder(x)

            h_x = self.layer_ATAC(x)
            h = self.shareEncoder(h_x)

            z, mu, logvar = self.sample(h)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                # recon_x = self.decoder(z)
                z = self.decoder_ATAC_hiden(z)
                if self.output_activation is not None:
                    recon_x = self.output_activation(self.reconstruction(z))
                else:
                    recon_x = self.reconstruction(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)
            elif out == 'h': # share out
                output.append(h.detach().cpu())

        output = torch.cat(output).numpy()

        return output



    def fit(self, dataloader_x, dataloader_y, mode, lr=0.0001, max_iter=100,
            alfa = 1, beda=1, delta=1, epsi=1, sita=1, save_path = './'):

        self.train().to(self.device)
        
        # weight_decay=5e-4
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)

        # n_epoch = int(np.ceil(max_iter/len(dataloader)))
        n_epoch = max_iter
        for epoch in tqdm(range(n_epoch)):
            loss_val = 0
            epoch_dl_loss, epoch_Diff_loss, epoch_share_loss = 0, 0, 0
            epoch_x_loss, epoch_kl_loss = 0, 0
            epoch_recon_loss, epoch_share_recon_loss, epoch_share_recon_loss = 0, 0, 0
            for _, ((x_batch, x_raw_batch, sf_batch, xl_batch),(y_batch, yl_batch)) in enumerate(zip(dataloader_x, dataloader_y)):
                
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                xl_tensor = Variable(xl_batch).to(self.device)
                
                y_tensor = Variable(y_batch).to(self.device)
                yl_tensor = Variable(yl_batch).to(self.device)

                optimizer.zero_grad()
                
                if self.cell_type == "scRNA":
                    # x_loss = self.LossscRNA(x_tensor, x_raw_tensor, sf_tensor)
                    loss = self.LossscRNA(x_tensor, x_raw_tensor, sf_tensor)

                elif self.cell_type == "scATAC":
                    y_recon_loss, y_kl_loss = self.LossscATAC(y_tensor)
                    loss = (y_recon_loss + y_kl_loss)/len(y_batch)
                
                elif self.cell_type == "scRNA_scATAC":

                    dl_loss = self.LossDomainLabel(x_tensor, y_tensor, xl_tensor, yl_tensor, mode)

                    Diff_loss = self.LossDiff(x_tensor, y_tensor, mode)

                    if mode == 'source':
                        share_loss = self.LossShare(x_tensor, x_raw_tensor, sf_tensor, y_tensor, mode)
                        x_loss = self.LossscRNA(x_tensor, x_raw_tensor, sf_tensor)

                        loss = alfa*dl_loss + beda*share_loss + epsi*x_loss + sita*Diff_loss
                        # loss = alfa*dl_loss + beda*share_loss + sita*Diff_loss

                    elif mode == 'target':
                        share_recon_loss, share_kl_loss = self.LossShare(x_tensor, x_raw_tensor, sf_tensor, y_tensor, mode)
                        y_recon_loss, y_kl_loss = self.LossscATAC(y_tensor)

                        loss = alfa*dl_loss + beda*(share_recon_loss + share_kl_loss)/len(y_batch) + delta*(y_recon_loss + y_kl_loss)/len(y_batch) + sita*Diff_loss

                        # loss = alfa*dl_loss/len(y_batch) + beda*(share_recon_loss + share_kl_loss)/len(y_batch) + sita*Diff_loss/len(y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10) # clip
                optimizer.step()
                
                loss_val += loss.item()
                epoch_dl_loss += dl_loss.item()
                epoch_Diff_loss += Diff_loss.item()
                if mode == 'source':
                    epoch_share_loss += share_loss.item()
                    # epoch_x_loss += x_loss.item()
                
                if mode == 'target':
                    epoch_share_recon_loss += share_recon_loss.item() 
                    epoch_share_recon_loss += share_kl_loss.item()
                    # epoch_recon_loss += y_recon_loss.item()
                    # epoch_kl_loss += y_kl_loss.item()
                    
            if epoch % 100 == 0:
                print('loss_val={:.3f} dl_loss={:.3f} Diff_loss={:.3f} '.format(
                    loss_val/len(x_tensor), epoch_dl_loss/len(x_tensor), epoch_Diff_loss/len(x_tensor)))
        # save model
        torch.save(self.state_dict(), save_path+'Embeding.pt')

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), self.n_centroids)
        pi = self.pi.repeat(N, 1) # NxK
#         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) + 1e-8 # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        z = self.EmbedingSharescATAC(dataloader, device)

        gmm.fit(z)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
