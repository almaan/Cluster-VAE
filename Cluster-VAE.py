#!/usr/bin/env python3

import torch as t
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
#%%
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import re

import argparse as arp
import datetime
import os.path as osp
import os

device = t.device('cpu')
plt.style.use('dark_background')

#%% Functions and classes -------------------
class SingleCellDataset(Dataset):
    def __init__(self,
                 sc_mat : pd.DataFrame ,
                 n_genes : int = -1):
        
        
        self.sc_mat = sc_mat
        
        self.sc_mat = self.sc_mat.iloc[:,(self.sc_mat.values == 0).sum(axis=0) > 0.05*self.sc_mat.shape[0]]
        self.sc_mat = self.sc_mat.iloc[self.sc_mat.values.sum(axis=1) > 100,:]
        
        self.G = (n_genes if (n_genes > 0) & (n_genes < self.sc_mat.shape[1])\
                  else self.sc_mat.shape[1])
        
        self.Grt = int(np.sqrt(self.G))
        
        self.cell_idx = self.sc_mat.index.tolist()
        self.gene_names = self.sc_mat.columns.tolist()
        
        self.sc_mat = self.sc_mat.values.astype(np.float32)
        self.sc_mat = np.log(self.sc_mat + 1.0)
#        self.sc_mat = self.sc_mat / self.sc_mat.max(axis = 1).reshape(-1,1)
        
        if self.G < self.sc_mat.shape[1]:
            srtidx = np.argsort(self.sc_mat.sum(axis=0))[::-1]
            self.sc_mat = self.sc_mat[:,srtidx[0:self.G]]
            self.gene_names = [self.gene_names[x] for x in srtidx[0:self.G]]
            
        self.data_len = len(self.cell_idx)
        
    def getimage(self,index):
        return (self.gene_names, self.__getitem__(index)[0].numpy())
    
    def __len__(self,):
        return self.data_len
    
    def __getitem__(self,index):
        name = self.cell_idx[index]
        counts = t.tensor(self.sc_mat[index,:])
        return (counts,name)

class VAE(nn.Module):
    def __init__(self,n_genes, latent_dim):
        
        super(VAE,self).__init__()
        
        self.G = n_genes
        self.layersize = [64, 32]
        self.latentsize = latent_dim
        
        # encoder variables 
        self.fc01 = nn.Linear(self.G, self.layersize[0])
        self.fc12 = nn.Linear(self.layersize[0],self.layersize[1])
        
        self.fc2zmu = nn.Linear(self.layersize[1],self.latentsize)
        self.fc2zlvar = nn.Linear(self.layersize[1],self.latentsize)
        
        # decoder variables
        self.fcz2 = nn.Linear(self.latentsize,self.layersize[1])
        self.fc21 = nn.Linear(self.layersize[1],self.layersize[0])
        self.fc10 = nn.Linear(self.layersize[0],self.G)
        
    def encode(self,x):
        h1 = nn.functional.relu(self.fc01(x))
        h2 = nn.functional.relu(self.fc12(h1))
        z_mu = self.fc2zmu(h2)
        z_lvar = self.fc2zlvar(h2)
        return z_mu, z_lvar
    
    def reparam(self, mu, lvar):
        std = t.exp(0.5*lvar)
        eps = t.randn_like(std)
        return mu + eps*std
    
    def decode(self,z):
        h4 = nn.functional.relu(self.fcz2(z))
        h5 = nn.functional.relu(self.fc21(h4))
#        x_hat = t.exp(self.fc10(h5))
        x_hat = nn.functional.relu(self.fc10(h5))
        return x_hat
    
    def forward(self,x):
        mu, lvar = self.encode(x.view(-1,n_genes))
        z = self.reparam(mu,lvar)
        return self.decode(z), mu, lvar


def loss_function(x_hat, x, mu, lvar, n_genes):
    bce = nn.functional.mse_loss(x_hat, x.view(-1,n_genes), reduction = 'sum')
    kld = -0.5 * t.sum(1 + lvar - mu.pow(2) - lvar.exp())
    return kld + bce


def train(epoch, optimizer):
    model.train()
    train_loss = 0.0
    for bix, (data,_) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        b_hat, mu, lvar = model(data)
        loss = loss_function(b_hat, data, mu, lvar, n_genes)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if bix % 10 == 0:
            btch = bix*int(data.shape[0])
            print(f"\rTrain Epoch : {epoch:10d} [{btch:10} / {len(train_loader.dataset)} ] | Loss : {loss.item():8e}",end='')
        
def fit(train_loader, model, nepochs, ngenes):
    
    optimizer = optim.Adam(model.parameters())
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    print('initiate training')
    try:
        for epoch in range(1,nepochs):
            train(epoch, optimizer)
        
            if epoch % 250 == 0 and args.graphical:
                print("Saving Comparision")
                with torch.no_grad():
                    ridx = np.random.randint(0,len(train_loader.dataset))
                    
                    names, y = train_loader.dataset.getimage(ridx)
                    y_hat,_,_ = model.forward(train_loader.dataset.__getitem__(ridx)[0])
                    y_hat = y_hat.numpy().reshape(-1,1)
                    ax.plot(y, linestyle = '-' , linewidth = 2)
                    ax.plot(y_hat, linestyle = '--', color = 'r', linewidth = 1)
                    ax.set_xticklabels(names, rotation = 45, fontsize = 10)
                    
                    fig.savefig(osp.join(gdir,str(epoch) + '.png'))
                    ax.clear()
    except KeyboardInterrupt:
        print('Early Interruption')
#%% Parser ----------------------------------
prs = arp.ArgumentParser()

prs.add_argument('-i','--input',
                 type = str,
                 required = True,
                 help = (''))

prs.add_argument('-o','--outdir',
                 type = str,
                 required = False,
                 default = '',
                 help = (''))

prs.add_argument('-g','--graphical',
                 action = 'store_true',
                 default = False,
                 help = (''))

prs.add_argument('-e','--epochs',
                 type = int,
                 default = 10000,
                 required = False,
                 help  = (''))

prs.add_argument('-b', '--batch_size',
                 type = int,
                 default = -1,
                 required = False,
                 help = (''))

prs.add_argument('-n','--ngenes',
                 type = int,
                 required = False,
                 default = -1,
                 help = (''))

prs.add_argument('-z','--latent_dim',
                 type = int,
                 required = False,
                 default = 10,
                 help = (''))

prs.add_argument('-k','--nclusters',
                 type = int,
                 required = False,
                 default = 3,
                 help = (''))

args = prs.parse_args()

tag = '.'.join([str(datetime.datetime.today()).replace(' ','').replace(':','.'),str(int(np.random.random()*1e5))])
mpth = args.input
outdir = ( args.outdir if args.outdir else osp.dirname(args.input))
outdir =  osp.join(outdir, ''.join(['results_',tag]))
if not osp.isdir(outdir): os.mkdir(outdir)

if args.graphical:
        gdir = osp.join(outdir,'predictions')
        if not osp.isdir(gdir): os.mkdir(gdir) 

n_genes = args.ngenes
scdata =  pd.read_csv(mpth,
                      sep = '\t',
                      index_col = 0,
                      header=0)

keep = [ re.search('^RP|MALAT1|^mt\.|^Rp|^rp',x) is None for x in scdata.columns]
scdata = scdata.loc[:,keep]
batch_size = (args.batch_size if (args.batch_size > 0) and \
              (args.batch_size < scdata.shape[0]) else scdata.shape[0])

train_loader = torch.utils.data.DataLoader(dataset= SingleCellDataset(scdata,n_genes = n_genes),
                                            batch_size = batch_size,
                                            shuffle = True)

model = VAE(n_genes, latent_dim=args.latent_dim).to(device)
fit(train_loader,
    model,
    args.epochs,
    n_genes)

#%% Visualization ------------------------------
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans

params = {'edgecolor' : 'white'}
clustering = BayesianGaussianMixture(n_components = args.nclusters,
                                     covariance_type='diag',
                                     max_iter= 1000,
                                     weight_concentration_prior_type = 'dirichlet_process')

dimred = TSNE(n_components = 2)
fig2, ax2 = plt.subplots(1,1)
cmap = iter([plt.cm.tab20(x) for x in range(0,20)])

with torch.no_grad():
    diter = iter(train_loader)
    y,lab = diter.next()
    mu, lvar = model.encode(y.view(-1,n_genes))
    y2 = model.reparam(mu,lvar)
    clustering.fit(y2.numpy())
    idx = clustering.fit_predict(y2.numpy())
#    idx = km.fit_predict(y2.numpy())
    dr = dimred.fit_transform(y2.numpy())
    mx,mn = np.max(dr), np.min(dr)
    
    ax2.set_xlim([mn,mx]) 
    
    for ii in np.unique(idx):
        clr = np.array(next(cmap)).reshape(1,-1)
        ax2.scatter(dr[idx == ii,0],dr[idx == ii,1], c = clr, **params) 

fig2.tight_layout()
fig2.savefig(osp.join(gdir,''.join([tag,'_tsne.png'])))
inlatent = pd.DataFrame(y2.numpy(),
                        index = lab,
                        columns = [str(x) for x in range(args.latent_dim)])

inlatent.to_csv(osp.join(outdir,''.join(['Z_',tag,'.tsv'])), sep = '\t', index = True, header = True)

#%% Save Results ------------------------------
obase = '.'.join(osp.basename(args.input).split('.')[0:-1])
ofile_meta = osp.join(outdir,''.join([obase,'.meta_',tag,'.tsv']))
ofile_cnt = osp.join(outdir,''.join([obase,'.cnt_',tag,'.tsv']))

meta = pd.DataFrame(index = lab)
meta['celltype'] = idx
meta['bio_celltype'] = idx
meta = meta[['celltype','bio_celltype']]

mm = scdata.loc[lab,train_loader.dataset.gene_names]

mm.to_csv(ofile_cnt,
          sep = "\t",
          index = True,
          header = True)

meta.to_csv(ofile_meta,
            sep = "\t",
            index = True,
            header = True)
