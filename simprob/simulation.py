"""
An abstraction for simulations of discrete-time processes.

Suitable for applying Kalman filters and other models.
"""

import dataclasses
import typing


class Mask(typing.Protocol):
    def __and__(self, other): ...


class NoObservation(Mask):
    def __and__(self, other):
        return other


Transition = typing.TypeVar("Transition", bound=typing.Callable)


@dataclasses.dataclass
class Iteration(typing.Generic[Transition]):
    transition: Transition
    observation: Mask = NoObservation()


def simulate(state, steps: typing.Iterable[Iteration]) -> typing.Iterator:
    yield state
    for step in steps:
        state = step.observation & step.transition(state)
        yield state
