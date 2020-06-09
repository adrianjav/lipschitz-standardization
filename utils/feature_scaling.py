from functools import wraps, partial, reduce

import numpy as np

import torch
import torch.optim
from torch.autograd import grad

import utils.distributions as my_dists
from .probabilistc_model import ProbabilisticModel


def normalize_per_dimension(prob_model, dims, func, name):
    if dims == 'all':
        dims = range(len(prob_model))
    elif dims == 'continuous':
        dims = [idxs for idxs, dist in prob_model.gathered if dist.real_dist.is_continuous]
        if len(dims) > 0:
            dims = reduce(list.__add__, dims)

    @wraps(func)
    def normalize_per_dimension_(x, mask=None):
        if len(dims) > 0:
            print('method:', name)

        for i in dims:
            dist_i = prob_model[i] if isinstance(prob_model, ProbabilisticModel) else prob_model[i][1]

            if dist_i.is_discrete:
                continue

            data = x[:, i] if (mask is None or mask[:, i].all()) else torch.masked_select(x[:, i], mask[:, i])
            data = dist_i >> data

            if isinstance(dist_i, my_dists.LogNormal):  # special case
                data = data.log()

            weight = func(data)
            dist_i.weight *= weight

            print(f'normalizing [dim={i}] [weight={dist_i.weight.item()}]')

        if len(dims) > 0:
            print('')

        return x
    return normalize_per_dimension_


normalize = partial(normalize_per_dimension, name='normalization', func=lambda x: 1/torch.max(torch.abs(x)).item())
standardize = partial(normalize_per_dimension, name='standardization', func=lambda x: 1/torch.std(x).item())

# Lipschitz standardization


def lipschitz_estimator(prob_model, data, mask):
    lipschitz = []
    for i, dist in enumerate(prob_model):
        if dist.is_discrete:
            lipschitz.append(None)
            continue

        pos = prob_model.gathered_index(i)
        if mask is None or mask[:, pos].all():
            new_data = data[..., i]
        else:
            new_data = torch.masked_select(data[..., i], mask[:, pos])

        # Compute second derivative of log wrt eta_i
        params = torch.stack(dist.params_from_data(new_data))
        params.requires_grad = True

        log_prob = dist.log_prob(new_data, params).mean(dim=0)  # for the exponential family we can take mean/sum
        g_log_prob = grad(log_prob, params, retain_graph=True, create_graph=True)[0]

        gg_log_prob = []
        for i in range(len(g_log_prob)):
            gg_log_prob.append(grad(g_log_prob[i], params, retain_graph=True, create_graph=True)[0].norm(p=1))
        gg_log_prob = torch.stack(gg_log_prob)

        lipschitz.append(gg_log_prob.detach().abs())

    return lipschitz


def weight_normal(values, goal):
    assert len(values) == 2, len(values)
    assert (values[1] > 0).all(), 'the second derivative wrt to eta2 should be positive'

    coeff1 = [values[1], values[0] + values[1], values[0], 0, -goal]
    roots1 = np.roots(coeff1)

    any = np.isreal(roots1) & (roots1 > 0)
    assert any.sum() < 2
    if any.sum() == 1:
        return torch.tensor(np.real(roots1[any]).item())

    raise Exception('Unique weight not found')  # This should never happen


def weight_gamma(values, goal):
    assert len(values) == 2
    assert (values[0] < goal).all(), 'all the second derivatives wrt eta1 should be smaller than the goal'
    assert (values[1] > 0).all(), 'the second derivative wrt to eta2 should be positive'

    coeff1 = [values[1], values[0] + values[1], values[0] - goal]
    roots1 = np.roots(coeff1)

    any = np.isreal(roots1) & (roots1 > 0)
    assert any.sum() < 2
    if any.sum() == 1:
        return torch.tensor(np.real(roots1[any]).item())

    raise Exception('Unique weight not found')  # This should never happen


def weight_exponential(values, goal):
    assert len(values) == 1
    assert (values[0] > 0).all(), 'all second derivatives should be positive'

    return torch.sqrt(goal / values[0])


def lipschitz_standardize(prob_model, dims, goal):
    if dims == 'all':
        dims = range(len(prob_model))

    proportion, size = [], sum([1 for _, d in prob_model.gathered if d.is_continuous])

    for i, [idxs, d] in enumerate(prob_model.gathered):
        if d.is_continuous:
            proportion += [1 / (size * len(idxs))] * len(idxs)
        else:
            proportion += [0] * len(idxs)

    def lipschitz_standardize_(x, mask=None):
        if len(dims) > 0:
            print('method:', 'Lipschitz-standardization')

        lipschitzs = lipschitz_estimator(prob_model, prob_model >> x, mask)

        for i in dims:
            dist_i = prob_model[i] if isinstance(prob_model, ProbabilisticModel) else prob_model[i][1]

            if dist_i.is_discrete:
                continue

            if isinstance(dist_i, my_dists.Normal) or isinstance(dist_i, my_dists.LogNormal):
                weight = weight_normal(lipschitzs[i], goal * proportion[i])
            elif isinstance(dist_i, my_dists.Gamma):
                weight = weight_gamma(lipschitzs[i], goal * proportion[i])
            elif isinstance(dist_i, my_dists.Exponential):
                weight = weight_exponential(lipschitzs[i], goal * proportion[i])
            else:
                raise ValueError(f'Wrong distribution: {dist_i}')

            dist_i.weight *= weight.unsqueeze(0)
            print(f'normalizing [dim={i}] [weight={dist_i.weight.item()}]')

        if len(dims) > 0:
            print('')

        return x

    return lipschitz_standardize_


def get_lipschitz(prob_model, data, mask=None):
    lipschitzs = lipschitz_estimator(prob_model, data, mask)
    values = []

    for i, dist in enumerate(prob_model):
        value = None
        if dist.is_continuous:
            f = torch.cat([f(dist.weight) for f in dist.f])
            value = f * f.norm(p=1) * lipschitzs[i]
            value = value.detach().tolist()

        values.append(value)

    return values
