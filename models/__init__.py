from .mixture import MixtureModel
from .matrix_factorization import MatrixFactorizationModel
from .vae import VAE


def create_model(model_name, prob_model, dataset, args):
    if model_name == 'mm':
        return MixtureModel(prob_model, args.num_clusters, dataset, args.print_every, empirical_init=False)
    elif model_name == 'mf':
        return MatrixFactorizationModel(prob_model, args.latent_size, dataset, args.print_every, empirical_init=False)
    elif model_name == 'vae':
        return VAE(prob_model, args.latent_size, args.hidden_size, args.print_every)

    raise AssertionError(f'Model not found: {model_name}')


__all__ = ['create_model']
