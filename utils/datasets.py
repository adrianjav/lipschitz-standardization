import math
import subprocess

import torch
import torch.distributions
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs

from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader

import pandas as pd


def get_dataloader(args, prob_model, preprocess_fn):
    dataset = RealWorldDataset(path=args.dataset, missing_percent=args.miss_perc, mask_suffix=args.miss_suffix,
                               categoricals=args.categoricals, prob_model=prob_model, num_clusters=args.num_clusters,
                               latent_size=args.latent_size, preprocess_fn=preprocess_fn)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    return loader


def read_data_file(filename, categoricals, prob_model):
    result = []
    df = pd.read_csv(filename, ',', header=None)

    for i in categoricals:
        df[i] -= df[i].min()

    for i in df:
        # assert not math.isnan(df[i][0])  # we assume that the first element is not nan
        df[i] = df[i].fillna(df[i].max())

        if prob_model.gathered[i][1].real_dist.dist.support == constraints.positive:
            df[i] = df[i].astype('float64').clip(lower=1e-30)  # ensure that is positive

    for _, line in df.iterrows():
        v = torch.tensor(line.tolist(), dtype=torch.float64)
        result += [v]

    original_data = torch.stack(result, dim=0)
    data = torch.stack(result, dim=0)

    return data, original_data


def read_mask_file(filename, nrows, ncols):
    df = pd.read_csv(filename, ',', names=['row', 'col'], header=None)
    df -= 1
    df = df.groupby('row')['col'].apply(list)

    mask = []
    for i in range(nrows):
        row = torch.tensor(df[i] if i in df.index else []).long()

        mask_i = torch.ones(ncols)
        mask_i.index_fill_(0, row, 0.)

        mask += [mask_i]

    return torch.stack(mask, dim=0)


def broadcast_mask(mask, prob_model):
    if all([d.num_dists == 1 for _, d in prob_model.gathered]):
        return mask

    new_mask = []
    for i, [idxs, _] in enumerate(prob_model.gathered):
        new_mask.append(mask[:, i].unsqueeze(-1).expand(-1, len(idxs)))

    return torch.cat(new_mask, dim=-1)


class RealWorldDataset(Dataset, torch.nn.Module):
    def __init__(self, path, missing_percent, mask_suffix, categoricals, prob_model, num_clusters=None,
                 latent_size=None, preprocess_fn=()):
        super().__init__()

        self.data, self.original_data = read_data_file(f'{path}/data.csv', categoricals, prob_model)
        self.ncols = self.original_data.size(1)
        self.num_clusters = num_clusters
        self.latent_size = latent_size
        self.which = path[path.rindex('/')+1:]
        self.is_cluster = num_clusters is not None

        mask = f'{path}/Missing{missing_percent}_{mask_suffix}.csv'
        mask = read_mask_file(mask, self.data.size(0), self.ncols)

        proc = subprocess.run(['ls', f'{path}/MissingTrue.csv'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0:
            nan_mask = read_mask_file(f'{path}/MissingTrue.csv', self.data.size(0), self.ncols)
        else:
            nan_mask = torch.ones_like(mask, dtype=torch.float64)

        self.mask = (mask.long() + nan_mask.long()) == 2
        self.missing_mask = ((1 - mask.long()) + nan_mask.long()) == 2

        self.mask_bc = broadcast_mask(self.mask, prob_model)
        self.missing_mask_bc = broadcast_mask(self.missing_mask, prob_model)

        for fn in preprocess_fn:
            self.data = fn(self.data, self.mask_bc)

        if latent_size is not None:
            self.params = Parameter(torch.empty((len(self.data), latent_size)).uniform_(-1, 1))
        elif num_clusters is not None:
            probs_z = torch.rand(len(self.data), num_clusters)
            probs_z = probs_z / probs_z.sum(dim=1, keepdim=True)
            self.params = Parameter(probs_to_logits(probs_z, is_binary=self.num_clusters==2))

    @property
    @torch.no_grad()
    def local_params(self):
        return logits_to_probs(self.params, is_binary=self.num_clusters==2) if self.is_cluster else self.params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], [self.original_data[idx], self.params[idx], self.mask_bc[idx]]

    def __str__(self):
        return f'{self.which}'

