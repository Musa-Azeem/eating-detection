from lib.data.datasets import WindowedDatasetWithStrideAndModeOfLabel
from lib.models import DAE
import torch
from pathlib import Path
import json
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def load_nursing_interpolated(winsize, batch_size, stride, split, model_path, k, device='cuda:0'):
    not_labeled = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    model_path = Path(model_path)
    config = json.load(open(model_path / 'config.json', 'r'))
    model = DAE(CONFIG=config).to(device)
    model.load_state_dict(torch.load(model_path / 'best_model.pt'))

    train_idx, dev_idx = split
    if set(train_idx).intersection(set(dev_idx)):
        raise ValueError(f"Train and dev indexes overlap")
    if set.intersection(set(train_idx).union(set(dev_idx)), not_labeled):
            raise ValueError(f"Some indexes are not labled")
    
    train_dataset = ConcatDataset([WindowedDatasetWithStrideAndModeOfLabel(nurse=idx,windowsize=winsize,stride=stride) for idx in train_idx])
    dev_dataset = ConcatDataset([WindowedDatasetWithStrideAndModeOfLabel(nurse=idx,windowsize=winsize,stride=stride) for idx in dev_idx])

    # Interpolate traindata
    trainloader_tmp = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    X_train, y_train = interpolate_embeddings(winsize, device, model, k, trainloader_tmp)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    devloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, devloader

def interpolate_embeddings(winsize, device, model, k, dataloader):
    model.eval()
    with torch.no_grad():
        embedding = []
        ys = []
        Xs = []
        for X,y in dataloader:
            Xs.append(X)
            ys.append(y)
            X = X.to(device)
            x = model.e(X)
            embedding.append(x.detach().cpu())
        X = torch.cat(Xs, dim=0)
        y = torch.cat(ys, dim=0)
        embedding = torch.cat(embedding, dim=0)
        emb_mean = embedding.mean(dim=2)

        # For each sample, find k nearest neighbors of same class
        nearest_neighbors = []
        for classi in range(5):
            embi = emb_mean[y==classi]
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(embi)
            distancesi, indices = nbrs.kneighbors() # find for all embi - embi itself not included
            nearest_neighbors.append(indices)

        # Interpolate 10 data points for each current data point (one for each nearest neighbor)
        X_interpolated = []
        y_interpolated = []

        for classi in range(5):
            neighbors = nearest_neighbors[classi]
            yi = y[y==classi]
            embi = embedding[y==classi]
            embi_nbrs = embi[neighbors]

            cprimei = (embi_nbrs - embi.unsqueeze(1)) * 0.5 + embi.unsqueeze(1)
            cprimei = cprimei.flatten(0,1)
            cprimeloader = DataLoader(torch.utils.data.TensorDataset(cprimei), batch_size=128, shuffle=False)
            Xprimeis = []
            for cprime in cprimeloader:
                Xprimei = model.decoder(cprime[0].to(device)).cpu()
                Xprimeis.append(Xprimei)
            Xprimei = torch.cat(Xprimeis, dim=0)

            X_interpolated.append(Xprimei)
            y_interpolated.append(torch.full((Xprimei.shape[0],), classi, dtype=yi.dtype))

        X_interpolated = torch.cat([X, *X_interpolated], dim=0)
        y_interpolated = torch.cat([y, *y_interpolated], dim=0)

    return X_interpolated, y_interpolated