"""
Backtrack a simulated state backwards and combine estimates from forward and backwards simulations.

See https://en.wikipedia.org/wiki/Forward%2Dbackward_algorithm
"""

import operator
import typing

from . import simulation

A = typing.TypeVar("A")
B = typing.TypeVar("B")


class Transition(typing.Protocol[A, B]):
    def __call__(self, value: A) -> B: ...
    def inv(self, value: B) -> A: ...


def simulate_bwd(
    steps: typing.Reversible[simulation.Iteration[Transition]], last_state
) -> typing.Iterator:
    """
    Backtracking simulation.

    This results with state estimates by integrating information from future observations.
    When combined with a forward simulation it provides a smoother state estimate at each step.
    """
    state = last_state
    for step in reversed(steps):
        yield state
        state = step.transition.inv(step.observation & state)
    yield state


def forward_backward(
    init_state, steps: typing.Reversible[simulation.Iteration[Transition]], last_state
) -> typing.Iterator:
    "Simulate forward and backwards filtering and combine the results to a single estimate."
    return map(
        operator.and_,
        simulation.simulate(init_state, steps),
        reversed(list(simulate_bwd(steps, last_state))),
    )
