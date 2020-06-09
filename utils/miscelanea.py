import torch
import torch.distributions as dists

import utils.distributions as my_dists


def fix_seed(seed) -> None:
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()

    return x_one_hot


def get_distribution_by_name(name):
    return {'normal': dists.Normal, 'gamma': dists.Gamma, 'bernoulli': dists.Bernoulli,
            'categorical': dists.Categorical, 'lognormal': dists.LogNormal,
            'poisson': dists.Poisson, 'exponential': dists.Exponential}[name]


def print_epoch_value(engine, metrics, trainer, max_epochs, print_every, fn=lambda x: x):
    if trainer.state.epoch % print_every != 1:
        return

    msg = f'Epoch {trainer.state.epoch} of {max_epochs}:'

    for name in metrics:
        value = fn(engine.state.metrics[name])
        msg += ' {} {:.5f}' if isinstance(value, float) else ' {} {}'

        if isinstance(value, torch.Tensor):
            value = value.tolist()

        msg = msg.format(name, value)

    print(msg)


def calculate_mie(engine, model, prob_model, dataset):
    epoch = engine.state.epoch
    mean = lambda x: sum(x).item() / len(x)

    if epoch % (engine.state.max_epochs / 4) != 0:
        return

    data = dataset.original_data
    observed_mask = getattr(dataset, 'mask', torch.ones_like(dataset.data))
    missing_mask = getattr(dataset, 'missing_mask', torch.ones_like(dataset.data))

    try:
        if str(model)[:3] == 'VAE':  # I can't import models due to circular dependencies
            mask_bc = getattr(dataset, 'mask_bc', None)
            pred = model.impute(dataset.data, mask_bc)
        else:
            pred = model.impute(dataset.local_params)

    except Exception:
        pred = torch.ones_like(data) * float('nan')

    observed_error = imputation_error(prob_model, pred, data, observed_mask)

    nominal_error = [e for e, [_, d] in zip(observed_error, prob_model.gathered) if d.real_dist.is_discrete]
    nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
    numerical_error = [e for e, [_, d] in zip(observed_error, prob_model.gathered) if d.real_dist.is_continuous]
    numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

    print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] observed imputation error:')
    for i, error in enumerate(observed_error):
        print(f'[dim={i}] {error}')
    print('nominal  :', nominal_error)
    print('numerical:', numerical_error)
    print('total    :', mean(observed_error))
    print('')

    if missing_mask.any():
        missing_error = imputation_error(prob_model, pred, data, missing_mask)

        nominal_error = [e for e, [_, d] in zip(missing_error, prob_model.gathered) if d.real_dist.is_discrete]
        nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
        numerical_error = [e for e, [_, d] in zip(missing_error, prob_model.gathered) if d.real_dist.is_continuous]
        numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

        print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] missing imputation error:')
        for i, error in enumerate(missing_error):
            print(f'[dim={i}] {error}')
        print('nominal  : ', nominal_error)
        print('numerical: ', numerical_error)
        print('total: ', mean(missing_error))
        print('')


def nrmse(pred, target, mask):  # for numerical variables
    norm_term = torch.max(target) - torch.min(target)
    new_pred = torch.masked_select(pred, mask.bool())
    new_target = torch.masked_select(target, mask.bool())

    return torch.sqrt(torch.nn.functional.mse_loss(new_pred, new_target)) / norm_term


def accuracy(pred, target, mask):  # for categorical variables
    return torch.sum((pred != target).double() * mask) / mask.sum()


def displacement(pred, target, mask, size):  # for ordinal variables
    diff = (target - pred).abs() * mask / size
    return diff.sum() / mask.sum()


def imputation_error(prob_model, pred, target, mask):
    mask = mask.double()

    errors = []
    for i, [_, dist] in enumerate(prob_model.gathered):
        pos = prob_model.gathered_index(i)

        if isinstance(dist.real_dist, my_dists.Categorical) or isinstance(dist.real_dist, my_dists.Bernoulli):
            errors.append(accuracy(pred[:, i], target[:, i], mask[:, pos]))
        else:  # numerical
            errors.append(nrmse(pred[:, i], target[:, i], mask[:, pos]))

    return errors


@torch.no_grad()
def calculate_ll(engine, model, prob_model, dataset):
    epoch = engine.state.epoch
    mean = lambda x: sum(x).item() / len(x)

    if epoch % (engine.state.max_epochs / 4) != 0:
        return

    # data = dataset.original_data
    observed_mask = getattr(dataset, 'mask', torch.ones_like(dataset.original_data))
    observed_mask_bc = getattr(dataset, 'mask_bc', torch.ones_like(dataset.data))

    try:
        if str(model)[:3] == 'VAE':  # I can't import models due to circular dependencies
            observed_log_prob = model.log_likelihood_real(dataset.data, dataset.original_data, mask=observed_mask,
                                                          mask_bc=observed_mask_bc).mean(dim=0)
        else:
            observed_log_prob = model.log_likelihood_real(dataset.original_data, dataset.params,
                                                          mask=observed_mask).mean(dim=0)

    except Exception:
        observed_log_prob = torch.ones_like(dataset.original_data) * float('nan')

    nominal_error = [e for e, [_, d] in zip(observed_log_prob, prob_model.gathered) if d.real_dist.is_discrete]
    nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
    numerical_error = [e for e, [_, d] in zip(observed_log_prob, prob_model.gathered) if d.real_dist.is_continuous]
    numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

    print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] observed log-likelihood:')
    for i, error in enumerate(observed_log_prob):
        print(f'[dim={i}] {error}')
    print('nominal  :', nominal_error)
    print('numerical:', numerical_error)
    print('total    :', mean(observed_log_prob))
    print('')
