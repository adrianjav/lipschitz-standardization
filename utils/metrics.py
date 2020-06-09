from abc import ABCMeta, abstractmethod

import torch
from ignite.engine import Events


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class Metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(Metaclass, 'temporary_class', (), {})


class Metric(with_metaclass(ABCMeta, object)):
    def __init__(self, keys):
        self.keys = keys

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, old_value, value):
        pass

    @abstractmethod
    def compute(self, value):
        pass

    @abstractmethod
    def step(self):
        pass

    def started(self, engine):
        setattr(engine.state, 'registers', getattr(engine.state, 'registers', {}))
        setattr(engine.state, 'metrics', getattr(engine.state, 'metrics', {}))

        engine.state.metrics.update(dict.fromkeys(self.keys, self.reset()))

    @torch.no_grad()
    def iteration_completed(self, engine):
        self.step()
        for i, v in engine.state.registers.items():
            if i in self.keys:
                engine.state.metrics[i] = self.update(engine.state.metrics[i], v)

    def completed(self, engine):
        for i in self.keys:
            result = self.compute(engine.state.metrics[i])
            if torch.is_tensor(result) and len(result.shape) == 0:
                result = result.item()
            engine.state.metrics[i] = result

    def attach(self, engine, handler=None, *args, vars=None, **kwargs):
        if not engine.has_event_handler(self.completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)

        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)

        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)

        if handler is not None:
            vars = vars or self.keys
            engine.add_event_handler(Events.EPOCH_COMPLETED, handler, vars, *args, **kwargs)


class Average(Metric):
    def __init__(self, vars):
        super(Average, self).__init__(vars)
        self.total = 0

    def reset(self):
        self.total = 0
        return 0

    def step(self):
        self.total += 1

    def update(self, old_value, value):
        return old_value + value

    def compute(self, value):
        return value / self.total

