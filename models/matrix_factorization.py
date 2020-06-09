import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import kl_divergence, constraints
from torch.distributions.utils import logits_to_probs, probs_to_logits


def eta_to_phi(prob_model, params):
    theta_loc, theta_rho = [], []
    gamma_loc, gamma_rho = [], []

    pos = 0
    for d in prob_model:
        for j in range(d.num_params):
            values = params[pos]

            if len(values.size()) == 0:
                values = values.unsqueeze(-1)

            if d.arg_constraints[j] == constraints.simplex:
                values = probs_to_logits(values)

            for value in values.unbind(dim=-1):
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

                # print(f'initializing pos {pos} with {value}')
                if j == 0:
                    theta_loc.append(value)
                else:
                    gamma_loc.append(value)

                value = torch.pow(10, torch.round(torch.log10(torch.clamp(value.abs(), min=1e-15)) - 1.5)) * 5
                if value < 700:
                    value = torch.log(torch.clamp(torch.exp(value) - 1, min=1e-15))

                if j == 0:
                    theta_rho.append(value)
                else:
                    gamma_rho.append(value)

                # print(f'initializing pos {pos} with rho {value}')
            pos += 1

    theta_loc = torch.stack(theta_loc)
    theta_rho = torch.stack(theta_rho)
    if len(gamma_loc) > 0:
        gamma_loc = torch.stack(gamma_loc)
        gamma_rho = torch.stack(gamma_rho)
    else:
        gamma_loc, gamma_rho = None, None

    return [theta_loc, theta_rho], [gamma_loc, gamma_rho]


class MatrixFactorizationModel(nn.Module):
    def __init__(self, prob_model, latent_size, dataset, print_every, empirical_init=True):
        super().__init__()
        self.prob_model = prob_model
        self.latent_size = latent_size
        self.print_every = print_every

        self.dist_offset = [0]
        for d in self.prob_model:
            self.dist_offset += [self.dist_offset[-1] + d.num_params - 1]

        self.num_params = [0]
        for d in self.prob_model:
            self.num_params.append(self.num_params[-1] + d.size_params[0])

        self.register_buffer('prior_z_loc', torch.zeros(latent_size))
        self.register_buffer('prior_z_scale', torch.ones(latent_size))
        self.register_buffer('prior_theta_loc', torch.zeros(self.num_params[-1], latent_size))
        self.register_buffer('prior_theta_scale', torch.ones(self.num_params[-1], latent_size))
        self.register_buffer('prior_gamma_loc', torch.zeros(self.dist_offset[-1]))
        self.register_buffer('prior_gamma_scale', torch.ones(self.dist_offset[-1]))

        self.z_rho = nn.Parameter(torch.log(torch.exp(torch.empty(latent_size).uniform_(0, 25) - 1)))
        self.has_gamma = self.dist_offset[-1] > 0

        mask = getattr(dataset, 'mask', None)

        if empirical_init:
            # params from the scaled data since we are going to unscale it afterwards
            params = prob_model.params_from_data(prob_model >> dataset.data, mask)

            theta, gamma = eta_to_phi(prob_model, params)
            theta[0] = theta[0].unsqueeze(-1).expand(-1, latent_size).clone()
            theta[1] = theta[1].unsqueeze(-1).expand(-1, latent_size).clone()
        else:
            z_i = self.q_z(dataset.local_params).mean

            # batch size x num_params x num_clusters
            theta = torch.rand((self.num_params[-1], latent_size)) * 0.25  # uniform(-0.25, 0.25)
            theta = theta.unsqueeze(0).expand(len(dataset.data), -1, -1)
            theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)  # Theta x Z_i ; batch_size x num_params

            if self.has_gamma:
                gamma = torch.rand((self.dist_offset[-1], )) * 0.25  # uniform(-0.25, 0.25)
                gamma = gamma.unsqueeze(0).expand(len(dataset.data), -1)
            else:
                gamma = None

            params = []
            for idxs, dist in prob_model.gathered:
                params_i = self.get_params(theta, gamma, idxs).mean(dim=1)
                params.extend(dist.scale_params(params_i))

            theta, gamma = eta_to_phi(prob_model, params)
            theta[0] = theta[0].unsqueeze(-1).expand(-1, latent_size).clone()
            theta[1] = theta[1].unsqueeze(-1).expand(-1, latent_size).clone()

        self.theta_loc = nn.Parameter(theta[0])
        self.theta_rho = nn.Parameter(theta[1])

        if self.has_gamma:
            self.gamma_loc = nn.Parameter(gamma[0])
            self.gamma_rho = nn.Parameter(gamma[1])

    @property
    def theta_scale(self):
        return softplus(self.theta_rho)

    @property
    def gamma_scale(self):
        return softplus(self.gamma_rho)

    @property
    def z_scale(self):
        return softplus(self.z_rho)

    @property
    def prior_theta(self):
        return dists.Normal(self.prior_theta_loc, self.prior_theta_scale)

    @property
    def prior_gamma(self):
        return dists.Normal(self.prior_gamma_loc, self.prior_gamma_scale)

    @property
    def prior_z(self):
        return dists.Normal(loc=self.prior_z_loc, scale=self.prior_z_scale)

    @property
    def q_theta(self):
        return dists.Normal(self.theta_loc, self.theta_scale)

    @property
    def q_gamma(self):
        return dists.Normal(self.gamma_loc, self.gamma_scale)

    def q_z(self, params):
        return dists.Normal(params, self.z_scale)

    def unpack_params(self, dist_idx, theta, gamma):
        noise = 1e-15
        dist = self.prob_model[dist_idx]

        value = theta[..., self.num_params[dist_idx]: self.num_params[dist_idx+1]].squeeze(-1)
        if isinstance(dist.arg_constraints[0], constraints.greater_than):
            lower_bound = dist.arg_constraints[0].lower_bound
            value = lower_bound + noise + softplus(value)
        elif isinstance(dist.arg_constraints[0], constraints.less_than):
            upper_bound = dist.arg_constraints[0].upper_bound
            value = upper_bound - noise - softplus(value)
        elif dist.arg_constraints[0] == constraints.simplex:
            value = logits_to_probs(value)

        params = [value]
        for i, pos in enumerate(range(self.dist_offset[dist_idx], self.dist_offset[dist_idx+1])):
            value = gamma[..., pos: pos + dist.size_params[i+1]]
            value = value.squeeze(-1)

            if isinstance(dist.arg_constraints[i+1], constraints.greater_than):
                lower_bound = dist.arg_constraints[i+1].lower_bound
                value = lower_bound + noise + softplus(value)
            elif isinstance(dist.arg_constraints[i+1], constraints.less_than):
                upper_bound = dist.arg_constraints[i+1].upper_bound
                value = upper_bound - noise - softplus(value)
            elif dist.arg_constraints[i+1] == constraints.simplex:
                value = logits_to_probs(value)

            params.append(value)

        return torch.stack(params, dim=0)

    def get_params(self, theta, gamma, dist_idxs=None):
        dist_idxs = dist_idxs if dist_idxs is not None else range(len(self.prob_model))
        if isinstance(dist_idxs, int): dist_idxs = [dist_idxs]

        params = [self.unpack_params(i, theta, gamma) for i in dist_idxs]
        return torch.cat(params, dim=0)

    def forward(self, x, state, original_x, zi_params, mask_bc=None):
        new_x = self.prob_model >> x
        elbo = self.elbo(new_x, zi_params, state, mask_bc)

        if self.training and hasattr(state, 'registers') and state.epoch % self.print_every == 1:
            with torch.no_grad():
                log_prob = self.log_likelihood_real(original_x, zi_params, None).mean(dim=0)
                state.registers.update({'re': log_prob.sum()})
                state.registers.update(
                    {f're_{i}': l_i.item() for i, l_i in enumerate(log_prob)}
                )

        return -elbo

    @torch.no_grad()
    def generate_data(self, locals):
        z_i = self.q_z(locals).mean

        theta = self.q_theta.mean.unsqueeze(0).expand((locals.size(0), -1, -1))
        theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)

        if self.has_gamma:
            gamma = self.q_gamma.mean.unsqueeze(0).expand((locals.size(0), -1))
        else:
            gamma = None

        x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = self.get_params(theta, gamma, idxs)
            x.append(dist_i.sample(1, dist_i << params).double().flatten())

        return torch.stack(x, dim=-1)

    @torch.no_grad()
    def impute(self, locals):
        z_i = self.q_z(locals).mean

        theta = self.q_theta.mean.unsqueeze(0).expand((locals.size(0), -1, -1))
        theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)

        if self.has_gamma:
            gamma = self.q_gamma.mean.unsqueeze(0).expand((locals.size(0), -1))
        else:
            gamma = None

        x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = self.get_params(theta, gamma, idxs)
            x.append(dist_i.impute(dist_i << params).double().flatten())

        return torch.stack(x, dim=-1)

    def __str__(self):
        return f'matrix factorization (latent size={self.latent_size})'

    def log_likelihood(self, x, zi_params, state, mask=None):
        z_i = self.q_z(zi_params).rsample()  # batch_size x latent_size
        theta = self.q_theta.rsample()
        gamma = self.q_gamma.rsample() if self.has_gamma else None

        theta = theta.unsqueeze(0).expand(len(x), -1, -1)  # batch size x num_params x latent_size
        theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)
        if self.has_gamma:
            gamma = gamma.unsqueeze(0).expand(len(x), -1)  # batch size x self.dist_offset[-1]

        log_prob = []
        for i, dist_i in enumerate(self.prob_model):
            params = self.unpack_params(i, theta, gamma)

            log_prob_i = dist_i.log_prob(x[..., i], params)
            log_prob += [log_prob_i]

        log_prob = torch.stack(log_prob, dim=-1)  # batch_size x num_dimensions
        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

    def log_likelihood_real(self, x, zi_params, mask=None):
        z_i = self.q_z(zi_params).mean  # mean

        theta = self.theta_loc.unsqueeze(0).expand(len(x), -1, -1)  # batch size x num_params x num_clusters
        theta = torch.bmm(theta, z_i.unsqueeze(2)).flatten(start_dim=1)

        gamma = self.gamma_loc.unsqueeze(0).expand(len(x), -1) if self.has_gamma else None

        log_prob = []
        for i, [idxs, dist_i] in enumerate(self.prob_model.gathered):
            params = dist_i << self.get_params(theta, gamma, idxs)
            log_prob += [dist_i.real_log_prob(x[..., i], params)]

        log_prob = torch.stack(log_prob, dim=-1)  # batch_size x num_dimensions
        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

    def elbo(self, x, zi_params, state, mask=None):
        log_prob = self.log_likelihood(x, zi_params, state, mask)
        re = log_prob.sum(dim=0).sum(dim=0)

        kl_z = kl_divergence(self.q_z(zi_params), self.prior_z).sum()  # batch_size
        kl_theta = kl_divergence(self.q_theta, self.prior_theta).sum()  # n_clusters x n_params
        if self.has_gamma:
            kl_gamma = kl_divergence(self.q_gamma, self.prior_gamma).sum()  # n_clusters x self.dist_offset[-1]
            kl_theta += kl_gamma

        kl_theta = kl_theta / len(state.dataloader)

        af = getattr(state, 'annealing_factor', 1.)
        elbo = re - af * (kl_z + kl_theta)

        if self.training and hasattr(state, 'registers') and state.epoch % self.print_every == 1:
            state.registers.update({
                'elbo': elbo.item(), 'kl_z': kl_z.item(), 'kl_theta': kl_theta.item()
            })

        return elbo
