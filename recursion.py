"""Functions to help with applying functions recursively.
"""
from typing import Callable
from numbers import Number
from numpy.typing import ArrayLike, NDArray

import numpy as np

def recursion_count(
    x_0: ArrayLike,
    func: Callable[[Number], Number],
    threshold: Number,
    max_it: int
    ) -> NDArray[np.uint]:
    """Returns the number of recursive applications of `func` required to exceed `threshold`.

    For each element in `x_0`, func is applied recursively: func(x_0), func(func(x_0)), etc. until
    the magnitude of the result exceeds `threshold`. The number of "func" calls required to get to
    that point is the returned value for the respective element in `x_0`.

    Args:
        x_0 (ArrayLike):
            The initial values for the recursion. Can include complex numbered values.
        func (Callable[[ArrayLike], ArrayLike]):
            The function to apply recursively to x_0. This must effectively be an element-wise
            function if it accept arrays; the results for one element must be completely independent
            of any other element. This is not checked. If in doubt, use a unary function together
            with `numpy.vectorize`.
        threshold (Number):
            A divergence threshold to compare the magnitude of the iterated value against.
        max_it (int):
            The maximum number of iterations to apply before assuming sub-threshold convergence.
            Must be positive.

    Returns:
        NDArray[np.uint]:
            The number of recursive applications of `func` required to exceed `threshold`; with the
            same shape as `x_0`. If the maximum number of iterations (`max_it`) is reached, the
            value specified `max_it + 1`.
    """
    x_0 = np.asarray(x_0)

    it_nums = range(1, max_it+1)

    is_prev_div = np.zeros(x_0.shape, dtype=np.bool) # was there divergence in a previous iteration?
    is_diverged = np.zeros(x_0.shape, dtype=np.bool)
    count = np.full(x_0.shape, dtype=np.uint, fill_value=max_it+1)
    x_n = x_0
    for it_num in it_nums:
        x_n[~is_diverged] = func(x_n[~is_diverged])
        is_prev_div[~is_prev_div] = count[~is_prev_div] < it_num
        is_diverged[~is_prev_div] = np.abs(x_n[~is_prev_div]) > threshold
        count[is_diverged & ~is_prev_div] = it_num

    return count
