"""
Backtrack a simulated state backwards and combine estimates from forward and backwards simulations.

See https://en.wikipedia.org/wiki/Forward%2Dbackward_algorithm
"""

import functools
import operator
import typing

import simprob

A = typing.TypeVar("A")
B = typing.TypeVar("B")


class Step(typing.Protocol[A, B]):
    def __call__(self, value: A) -> B: ...
    def inv(self, value: B) -> A: ...


def simulate_bwd(steps: typing.Reversible[Step], last_state) -> typing.Iterator:
    """
    Backtracking simulation.

    This results with state estimates by integrating information from future observations.
    When combined with a forward simulation it provides a smoother state estimate at each step.
    """
    state = last_state
    for step in reversed(steps):
        yield state
        state = bwd_step(step, state)
    yield state


def forward_backward(
    init_state, steps: typing.Reversible[Step], last_state
) -> typing.Iterator:
    "Simulate forward and backwards filtering and combine the results to a single estimate."
    return map(
        operator.and_,
        simprob.simulate(init_state, steps),
        reversed(list(simulate_bwd(steps, last_state))),
    )


def bwd_step(step, state):
    "Apply step backwards, supports a few common operators (`+`, `&`, `@`)"
    if hasattr(step, "inv"):
        return step.inv(state)
    assert isinstance(step, functools.partial)
    return bwd_funcs[step.func](*step.args, state)


# matmul is deliberately absent,
# and is rather wrapped by kalman.KalmanTransition and hidden_markov.TransitionMatrix,
# because what they do are different (transforming state vectors vs transitioning between discrete states)
bwd_funcs = {
    operator.and_: operator.and_,
    operator.add: lambda x, y: y - x,
}
