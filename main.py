import os
import sys
import argparse
import subprocess
from datetime import datetime

import torch

import utils.plotting as plt
import utils.feature_scaling as scaling
from utils.datasets import get_dataloader
from models import create_model, VAE
from utils.trainer import create_trainer
from utils.probabilistc_model import ProbabilisticModel
from utils.miscelanea import fix_seed


def validate(args) -> None:
    args.timestamp = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    if args.dataset[-1] == '/':
        args.dataset = args.dataset[:-1]

    assert (args.trick is not None) + args.std_none + args.max == 1, 'trick, std-none and max are mutually exclusive.'

    if args.trick:
        args.lip_std = 'all'
    else:
        args.lip_std = []

    if args.trick == 'gamma':
        method = 'ours-gamma'
    elif args.trick == 'bern':
        method = 'ours-bern'
    elif args.trick == 'none':
        method = 'ours-none'

    if args.max:
        method = 'max'
        args.max = 'all'
    else:
        args.max = []

    if args.std_none:
        method = 'std-none'
        args.std_none = 'all'
    else:
        args.std_none = []

    args.root = f'{args.root}/{args.dataset}/{args.model}/{method}/' \
                f'Missing{args.miss_perc}_{args.miss_suffix}/seed_{args.seed}'

    if args.learning_rate is None:
        args.learning_rate = 1e-2 if args.model == 'mf' else 1e-3

    arguments = ['./read_types.sh', f'{args.dataset}/data_types.csv']
    if args.trick == 'gamma':
        arguments.append('--gamma-trick')
    elif args.trick == 'bern':
        arguments.append('--bern-trick')

    proc = subprocess.Popen(arguments, stdout=subprocess.PIPE)
    out = eval(proc.communicate()[0].decode('ascii'))

    args.probabilistic_model = out['probabilistic model']
    args.categoricals = out['categoricals']

    assert args.latent_size is None or args.num_clusters is None

    if args.max_epochs is None:
        args.max_epochs = {
            'Wine': 2000, 'letter': 400, 'spam': 2000,
            'Adult': 400, 'defaultCredit': 400, 'Breast': 3000
        }[args.dataset[args.dataset.rindex('/')+1:]]


def generate_preprocess_functions(prob_model, args):
    preprocess_fn = [
        prob_model.preprocess_data,
        scaling.standardize(prob_model, 'continuous')
    ]

    preprocess_fn += [scaling.normalize(prob_model, args.max)]
    preprocess_fn += [scaling.standardize(prob_model, args.std_none)]
    preprocess_fn += [scaling.lipschitz_standardize(prob_model, args.lip_std, 1 / args.learning_rate)]

    return preprocess_fn


def print_data_info(prob_model, data):
    print()
    print('#' * 20)
    print('Original data')

    x = data
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, i].min()} {x[:, i].max()}')

    print()
    print(f'weights = {[x.item() for x in prob_model.weights]}')
    print()

    print('Scaled data')

    x = prob_model >> data
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, i].min()} {x[:, i].max()}')

    print('#' * 20)
    print()


def print_info(model, prob_model, loader):
    mask = getattr(loader.dataset, 'mask', None)
    lips = scaling.get_lipschitz(prob_model, loader.dataset.data, mask)
    lips = [(sum(x) if x is not None else None) for x in lips]

    # print(f'weights      = {prob_model.weights}')
    print(f'Lipschitzs = {lips}')
    print()

    print('Dataset:', loader.dataset)
    print('Model:', model)


def train(trainer, loader, max_epochs):
    try:
        trainer.run(loader, max_epochs=max_epochs)

    except KeyboardInterrupt:
        from ignite.engine import Events
        trainer.fire_event(Events.COMPLETED)
        trainer.terminate()
        print(f'Training interrupted by keyboard.', file=sys.stderr)
        print('', file=sys.stderr)


@torch.no_grad()
def evaluate(model, prob_model, loader):
    if isinstance(model, VAE):
        mask_bc = getattr(loader.dataset, 'mask_bc', None)
        generated_data = model.generate_data(loader.dataset.data, mask_bc)
    else:
        generated_data = model.generate_data(loader.dataset.local_params)

    data = loader.dataset.original_data
    plt.plot_together([data, generated_data], prob_model, title='', legend=['original', 'generated'],
                      path=f'{args.root}/marginal')


def main(args):
    validate(args)
    fix_seed(args.seed)

    os.makedirs(args.root, exist_ok=True)

    if args.to_file:
        sys.stdout = open(f'{args.root}/stdout.txt', 'w')
        sys.stderr = open(f'{args.root}/stderr.txt', 'w')

    prob_model = ProbabilisticModel(args.probabilistic_model)
    print('Likelihoods:', [str(d) for d in prob_model])

    if args.latent_size is None:
        if args.model == 'vae':
            args.latent_size = max(1, int(len(prob_model.gathered) * 0.75 + 0.5))
        elif args.model == 'mf':
            args.latent_size = max(1, int(len(prob_model.gathered) * 0.5 + 0.5))

    if args.num_clusters is None and args.model == 'mm':
        dataset = args.dataset[args.dataset.rindex('/')+1:]
        args.num_clusters = 5 if dataset in ['Wine', 'Breast', 'spam'] else 10

    preprocess_fn = generate_preprocess_functions(prob_model, args)
    loader = get_dataloader(args, prob_model, preprocess_fn)

    model = create_model(args.model, prob_model, loader.dataset, args)

    print_data_info(prob_model, loader.dataset.data)
    print_info(model, prob_model, loader)

    trainer = create_trainer(model, prob_model, loader.dataset, args)
    train(trainer, loader, args.max_epochs)

    print_info(model, prob_model, loader)
    evaluate(model, prob_model, loader)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    # Configuration
    parser = argparse.ArgumentParser('')
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-root', type=str, default='results', help='Output folder (default: %(default)s)')
    parser.add_argument('-to-file', action='store_true', help='Redirect output to \'stdout.txt\'')
    parser.add_argument('-batch-size', type=int, default=1024, help='Batch size (default: %(default)s)')
    parser.add_argument('-learning-rate', type=float, default=None, help='Learning rate (default: 1e-2 if MF, '
                                                                         '1e-3 otherwise)')
    parser.add_argument('-max-epochs', type=int, default=None, help='Max epochs (default: as described in the appendix)')
    parser.add_argument('-print-every', type=int, default=25, help='Interval to print (default: %(default)s)')

    parser.add_argument('-model', type=str, required=True, choices=['mm', 'vae', 'mf'],
                        help='Model to use: Mixture Model (mm), Matrix Factorization (mf), or VAE (vae)')
    parser.add_argument('-latent-size', type=int, default=None)
    parser.add_argument('-num-clusters', type=int, default=None)
    parser.add_argument('-hidden-size', type=int, default=256, help='Size of the hidden layers (VAE)')

    parser.add_argument('-dataset', type=str, required=True, help='Dataset to use (path to folder)')
    parser.add_argument('-miss-perc', type=int, required=True, help='Missing percentage')
    parser.add_argument('-miss-suffix', type=int, required=True, help='Suffix of the missing percentage file')

    parser.add_argument('-trick', default=None, choices=['gamma', 'bern', 'none'], help='Trick to use (if any)')
    parser.add_argument('-max', action='store_true', help='Normalize data')
    parser.add_argument('-std-none', action='store_true', help='Standardize data')

    args = parser.parse_args()
    main(args)

    sys.exit(0)
