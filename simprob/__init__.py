import functools
import operator
import typing


def simulate(start, steps: typing.Iterable[typing.Callable]) -> typing.Iterator:
    state = start
    yield state
    for step in steps:
        state = step(state)
        yield state


def fuse(x):
    """
    fuse(x) is a convinience for functools.partial(operator.and_, x)

    It can be used for incorporating observations/state-measurements in the list of simulation steps.
    """
    return functools.partial(operator.and_, x)


def prepend_all(
    elems: typing.List[typing.Callable], iterator: typing.Iterable[typing.Callable]
) -> typing.List[typing.Callable]:
    """
    Convinience function to prepend list elements before every element in a given iterator.

    It can be used for adding a constant transition step between observation steps in a simulation.
    """
    return [x for cur in iterator for x in elems + [cur]]
