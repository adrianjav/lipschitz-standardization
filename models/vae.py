from functools import partial

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import kl_divergence, constraints
from torch.distributions.utils import logits_to_probs


class VAE(nn.Module):
    def __init__(self, prob_model, latent_size, hidden_size, print_every):
        super().__init__()
        self.prob_model = prob_model
        self.print_every = print_every

        self.dist_offset = [0]
        for d in self.prob_model:
            self.dist_offset += [self.dist_offset[-1] + d.num_params]
        self.num_params = sum([sum(d.size_params) for d in prob_model])

        # Prior
        self.register_buffer('prior_z_loc', torch.zeros(latent_size))
        self.register_buffer('prior_z_scale', torch.ones(latent_size))

        # Encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(len(prob_model)),
            nn.Linear(len(prob_model), hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

        self.encoder_loc = nn.Linear(hidden_size, latent_size)
        self.encoder_logscale = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, self.num_params)
        )

        def init_weights(m, gain=1.):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.01)

        self.encoder.apply(partial(init_weights, gain=nn.init.calculate_gain('tanh')))
        self.encoder_loc.apply(init_weights)
        self.encoder_logscale.apply(init_weights)
        self.decoder.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))

    @property
    def prior_z(self):
        return dists.Normal(self.prior_z_loc, self.prior_z_scale)

    def q_z(self, loc, logscale):
        scale = softplus(logscale)
        return dists.Normal(loc, scale)

    def encode(self, x, mask=None):
        h = self.encoder(x if mask is None else x * mask.double())

        # Normal distribution
        loc = self.encoder_loc(h)  # constraints.real
        logscale = self.encoder_logscale(h)  # constraints.real

        return loc, logscale

    def decode(self, z):
        return self.decoder(z)

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

    def get_params(self, dist_idxs, theta):
        if isinstance(dist_idxs, int):
            dist_idxs = [dist_idxs]

        params = [self.unpack_params(i, theta) for i in dist_idxs]
        return torch.cat(params, dim=0)

    def forward(self, x, state, original_x, zi_logits, mask_bc=None):
        z_params = self.encode(x, mask_bc)
        z = self.q_z(*z_params).rsample()
        theta = self.decode(z)  # batch_size x num_params

        log_prob = self.log_likelihood(self.prob_model >> x, theta, state, mask_bc)  # batch_size x D
        re = log_prob.sum(dim=0).sum(dim=0)

        kl_z = kl_divergence(self.q_z(*z_params), self.prior_z).sum()

        af = getattr(state, 'annealing_factor', 1.) if state is not None else 1.
        elbo = re - af * kl_z

        if self.training and hasattr(state, 'registers') and state.epoch % self.print_every == 1:
            state.registers.update({
                'elbo': elbo.item(), 'kl_z': kl_z.item()
            })

            with torch.no_grad():
                log_prob = self.log_likelihood_real(x, original_x, None, mask_bc).mean(dim=0)
                state.registers.update({'re': log_prob.sum()})
                state.registers.update(
                    {f're_{i}': l_i.item() for i, l_i in enumerate(log_prob)}
                )

        return -elbo

    @torch.no_grad()
    def generate_data(self, x, mask):  # It is not actually generating new data
        z_params = self.encode(x, mask)
        z = self.q_z(*z_params).sample()
        theta = self.decode(z)  # batch_size x num_params

        new_x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = self.get_params(idxs, theta=theta)
            new_x.append(dist_i.sample(1, dist_i << params).double().flatten())

        return torch.stack(new_x, dim=-1)

    @torch.no_grad()
    def impute(self, x, mask):
        z_params = self.encode(x, mask)
        z = self.q_z(*z_params).mean
        theta = self.decode(z)  # batch_size x num_params

        new_x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = self.get_params(idxs, theta=theta)
            new_x.append(dist_i.impute(dist_i << params).double().flatten())

        return torch.stack(new_x, dim=-1)

    # Measures
    def log_likelihood(self, x, theta, state, mask):
        log_prob = []
        for i, dist_i in enumerate(self.prob_model):
            params = self.unpack_params(i, theta)

            log_prob_i = dist_i.log_prob(x[..., i], params)
            log_prob.append(log_prob_i)

        log_prob = torch.stack(log_prob, dim=-1)  # batch_size x num_dimensions
        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob

    def log_likelihood_real(self, x, original_x, mask, mask_bc):
        z_params = self.encode(x, mask_bc)
        z = self.q_z(*z_params).sample()
        theta = self.decode(z)  # batch_size x num_params

        log_prob = []
        for i, [idxs, dist_i] in enumerate(self.prob_model.gathered):
            params = self.get_params(idxs, theta)
            log_prob_i = dist_i.real_log_prob(original_x[..., i], dist_i << params)
            log_prob.append(log_prob_i)

        log_prob = torch.stack(log_prob, dim=-1)  # batch_size x num_dimensions
        if mask is not None:
            log_prob = log_prob * mask.double()

        return log_prob
