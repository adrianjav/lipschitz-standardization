import math

from torch.optim import Adam

from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan, ModelCheckpoint

from utils.miscelanea import print_epoch_value, calculate_mie, calculate_ll
from .metrics import Average


def create_trainer(model, prob_model, dataset, args):
    trainer = build_trainer(model, dataset, args.learning_rate, args.max_epochs, args.root)

    # Per-epoch parameters
    @trainer.on(Events.EPOCH_STARTED)
    def compute_temperature(engine):
        r = 1e-3

        if (engine.state.epoch - 1) % 20 == 0:
            engine.state.temperature = max(r, math.exp(- r * engine.state.epoch))

    average = Average(['elbo', 're', 'kl_z', 'kl_theta'])
    average.attach(trainer, print_epoch_value, trainer, vars=['elbo', 're'], max_epochs=args.max_epochs,
                   print_every=args.print_every)

    errors = Average([f're_{i}' for i in range(len(prob_model.gathered))])
    errors.attach(trainer, print_epoch_value, trainer, max_epochs=args.max_epochs, print_every=args.print_every)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, calculate_mie, model, prob_model, dataset)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, calculate_ll, model, prob_model, dataset)

    return trainer


def build_trainer(model, dataset, learning_rate, max_epochs, root):
    optim = Adam([{'params': model.parameters()}, {'params': dataset.parameters()}], lr=learning_rate)

    def trainer_step(engine, batch):
        optim.zero_grad()

        x, y = batch

        loss = model(x, engine.state, *y)
        loss.backward()

        optim.step()

        return loss.item()

    trainer = Engine(trainer_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    handler = ModelCheckpoint(root, 'checkpoint', n_saved=1, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=max_epochs), handler, {'model': model})

    return trainer

