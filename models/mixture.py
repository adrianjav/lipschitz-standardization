import itertools

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import kl_divergence, constraints
from torch.distributions.utils import logits_to_probs, probs_to_logits

from utils.distributions import GumbelDistribution
from utils.miscelanea import to_one_hot


def eta_to_phi(prob_model, params, size):
    theta_loc, theta_rho = torch.empty([size]), torch.empty([size])

    offset, pos = 0, 0
    for d in prob_model:
        for j in range(d.num_params):
            values = params[pos]

            if len(values.size()) == 0:
                values = values.unsqueeze(-1)

            if d.arg_constraints[j] == constraints.simplex:
                values = probs_to_logits(values)

            offset -= 1
            for value in values.unbind(dim=-1):
                offset += 1

                if isinstance(d.arg_constraints[j], constraints.greater_than):
                    lower_bound = d.arg_constraints[j].lower_bound
                    if (value - lower_bound).abs() < 700:
                        value = torch.log(torch.clamp(torch.exp(value - lower_bound) - 1, min=1e-15))
                    else:
                        assert value > d.arg_constraints[j].lower_bound, value
                elif isinstance(d.arg_constraints[j], constraints.less_than):
                    upper_bound = d.arg_constraints[j].upper_bound
                    if (upper_bound - value).abs() < 700:
                        value = torch.log(torch.clamp(torch.exp(upper_bound - value) - 1, min=1e-15))
                    else:
                        assert value < d.arg_constraints[j].upper_bound, value
                theta_loc[pos + offset] = value

                value = torch.pow(10, torch.round(torch.log10(torch.clamp(value.abs(), min=1e-15)) - 3.5)) * 5
                if value < 700:
                    value = torch.log(torch.clamp(torch.exp(value) - 1, min=1e-15))
                theta_rho[pos + offset] = value

                # print(f'initializing pos {pos} with rho {value}')
            pos += 1

    return theta_loc, theta_rho


class MixtureModel(nn.Module):
    def __init__(self, prob_model, num_clusters, dataset, print_every, empirical_init=True):
        super(MixtureModel, self).__init__()
        self.prob_model = prob_model
        self.num_clusters = num_clusters
        self.print_every = print_every

        self.dist_offset = [0]
        for d in self.prob_model:
            self.dist_offset += [self.dist_offset[-1] + d.num_params]
        self.num_params = sum([sum(d.size_params) for d in prob_model])

        self.register_buffer('prior_z_pi', torch.ones(num_clusters) / num_clusters)
        self.register_buffer('prior_theta_loc', torch.zeros(self.num_params, num_clusters))
        self.register_buffer('prior_theta_scale', torch.ones(self.num_params, num_clusters))

        self.theta_loc = torch.Tensor(self.num_params, num_clusters).uniform_(-0.25, 0.25)
        self.theta_rho = torch.Tensor(self.num_params, num_clusters).uniform_(0, 0.5)
        self.theta_rho = torch.log(torch.exp(self.theta_rho) - 1)

        mask = getattr(dataset, 'mask', None)

        if empirical_init:
            # params from the scaled data since we are going to unscale it afterwards
            params = prob_model.params_from_data(prob_model >> dataset.data, mask)

            theta_loc, theta_rho = eta_to_phi(prob_model, params, self.num_params)
            theta_loc = theta_loc.unsqueeze(-1).expand(-1, num_clusters).clone()
            theta_rho = theta_rho.unsqueeze(-1).expand(-1, num_clusters).clone()
        else:
            theta_loc, theta_rho = [], []
            clusters = dists.Categorical(torch.ones(self.num_clusters)/self.num_clusters).sample([dataset.data.size(0)])

            for k in range(num_clusters):
                params_k = []
                for i, dist_i in enumerate(prob_model):
                    pos = prob_model.gathered_index(i)
                    if mask is None or mask[:, pos].all():
                        data = torch.masked_select(dataset.data[..., i], clusters == k)
                    else:
                        data = torch.masked_select(dataset.data[..., i], (clusters == k) & mask[:, pos])

                    params_k.extend(dist_i.params_from_data(dist_i >> data))

                theta_loc_k, theta_rho_k = eta_to_phi(prob_model, params_k, self.num_params)
                theta_loc.append(theta_loc_k)
                theta_rho.append(theta_rho_k)

            theta_loc, theta_rho = torch.stack(theta_loc, dim=-1), torch.stack(theta_rho, dim=-1)

        self.theta_loc = nn.Parameter(theta_loc)
        self.theta_rho = nn.Parameter(theta_rho)

        self.last_call = None

    @property
    def theta_scale(self):
        return softplus(self.theta_rho)

    @property
    def prior_theta(self):
        return dists.Normal(self.prior_theta_loc, self.prior_theta_scale)

    @property
    def prior_z(self):
        return dists.OneHotCategorical(probs=self.prior_z_pi)

    @property
    def q_theta(self):
        return dists.Normal(self.theta_loc, self.theta_scale)

    def q_z(self, logits=None, probs=None, temperature=1.):
        return GumbelDistribution(logits=logits, probs=probs, temperature=temperature)

    def unpack_params(self, dist_idx, theta):
        noise = 1e-15
        dist = self.prob_model[dist_idx]

        params = []
        for i, pos in enumerate(range(self.dist_offset[dist_idx], self.dist_offset[dist_idx+1])):
            value = theta[..., pos: pos + dist.size_params[i]]
            value = value.squeeze(-1)

            if isinstance(dist.arg_constraints[i], constraints.greater_than):
                lower_bound = dist.arg_constraints[i].lower_bound
                value = lower_bound + noise + softplus(value)

            elif isinstance(dist.arg_constraints[i], constraints.less_than):
                upper_bound = dist.arg_constraints[i].upper_bound
                value = upper_bound - noise - softplus(value)

            elif dist.arg_constraints[i] == constraints.simplex:
                value = logits_to_probs(value)

            params += [value]

        return torch.stack(params, dim=0)

    def get_params(self, dist_idxs=None, cluster_idx=None, theta=None):
        theta = theta if theta is not None else self.theta_loc

        if cluster_idx is not None: theta = theta[..., cluster_idx]

        dist_idxs = dist_idxs if dist_idxs is not None else range(len(self.prob_model))
        if isinstance(dist_idxs, int): dist_idxs = [dist_idxs]

        params = [self.unpack_params(i, theta) for i in dist_idxs]
        return torch.cat(params, dim=0)

    def forward(self, x, state, original_x, zi_logits, mask_bc=None):
        new_x = self.prob_model >> x
        elbo = self.elbo(new_x, zi_logits, state, mask_bc)

        if self.training and hasattr(state, 'registers') and state.epoch % self.print_every == 1:
            with torch.no_grad():
                log_prob = self.log_likelihood_real(original_x, zi_logits, None).mean(dim=0)
                state.registers.update({'re': log_prob.sum()})
                state.registers.update(
                    {f're_{i}': l_i.item() for i, l_i in enumerate(log_prob)}
                )

        return -elbo

    @torch.no_grad()
    def generate_data(self, locals):
        z_i = self.q_z(probs=locals.mean(dim=0)).sample([len(locals)])
        theta = self.q_theta.sample()

        theta = theta.unsqueeze(0).expand((z_i.size(0), -1, -1))
        theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)

        x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = self.get_params(idxs, theta=theta)
            x.append(dist_i.sample(1, dist_i << params).double().flatten())

        return torch.stack(x, dim=-1)

    @torch.no_grad()
    def impute(self, locals):
        z_i = locals.max(dim=1)[1]  # argmax
        z_i = to_one_hot(z_i, locals.size(1)).double()

        theta = self.q_theta.mean.unsqueeze(0).expand((z_i.size(0), -1, -1))
        theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)

        x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = self.get_params(idxs, theta=theta)
            x.append(dist_i.impute(dist_i << params).double().flatten())

        return torch.stack(x, dim=-1)

    def __str__(self):
        return f'mixture model (num clusters={self.num_clusters})'

    def log_likelihood(self, x, zi_logits, state, mask=None):
        temperature = state.temperature
        z_i = self.q_z(logits=zi_logits, temperature=temperature).rsample()  # batch_size x num_clusters
        theta = self.q_theta.rsample()

        theta = theta.unsqueeze(0).expand(len(x), -1, -1)  # batch size x num_params x num_clusters
        theta = torch.bmm(theta, z_i.unsqueeze(2)).squeeze(2)  # Theta x Z_i ; batch_size x num_params

        log_prob = []
        for i, dist_i in enumerate(self.prob_model):
            params = self.unpack_params(i, theta=theta)

            log_prob_i = dist_i.log_prob(x[..., i], params)
            log_prob += [log_prob_i]

        log_prob = torch.stack(log_prob, dim=-1)  # batch_size x num_dimensions
        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

    def log_likelihood_real(self, x, zi_logits, mask=None):
        z_i = self.q_z(logits=zi_logits).sample()  # batch_size x num_clusters

        theta = self.theta_loc.unsqueeze(0).expand(len(x), -1, -1)  # batch size x num_params x num_clusters
        theta = torch.bmm(theta, z_i.unsqueeze(2)).squeeze(2)  # Theta x Z_i ; batch_size x num_params

        log_prob = []
        for i, [idxs, dist_i] in enumerate(self.prob_model.gathered):
            params = dist_i << self.get_params(idxs, theta=theta)
            log_prob += [dist_i.real_log_prob(x[..., i], params)]

        log_prob = torch.stack(log_prob, dim=-1)  # batch_size x num_dimensions
        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

    def elbo(self, x, zi_logits, state, mask=None):
        log_prob = self.log_likelihood(x, zi_logits, state, mask)
        re = log_prob.sum(dim=0).sum(dim=0)
        kl_z = kl_divergence(dists.OneHotCategorical(logits=zi_logits), self.prior_z).sum(dim=0)  # batch_size
        kl_theta = kl_divergence(self.q_theta, self.prior_theta).sum()  # n_clusters x n_params

        kl_theta = kl_theta / len(state.dataloader)

        af = getattr(state, 'annealing_factor', 1.)
        elbo = re - af * (kl_z + kl_theta)

        if self.training and hasattr(state, 'registers') and state.epoch % self.print_every == 1:
            state.registers.update({
                'elbo': elbo.item(), 'kl_z': kl_z.item(), 'kl_theta': kl_theta.item()
            })

        return elbo
