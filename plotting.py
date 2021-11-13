""" Functions to help plot recursive map convergence in the complex plane (e.g. Julia sets).
"""
from typing import Callable, Tuple
from numbers import Number
from matplotlib.pyplot import Axes
from matplotlib.image import AxesImage

import numpy as np

from recursion import recursion_count, ArrayLike

def plot_recursions_convergence( #pylint: disable=invalid-name
    ax: Axes,
    func: Callable[[ArrayLike], ArrayLike],
    threshold: Number,
    max_it: int,
    extent: Tuple[Number, Number, Number, Number],
    pixels_per_axis: Number,
    **imshow_kwargs
    ) -> AxesImage:
    """Plots the complex plane coloured by the recursion level needed to diverge past `threshold`.

    These effective show the Julia sets of the given function, but with attractive colouring around
    the outsides.

    Args:
        ax (Axes):
            The `matplotlib` Axes object to plot on.
        func (Callable[[ArrayLike], ArrayLike]):
            The function to apply recursively to elements in the complex plane. This must
            effectively be an element-wise function if it accept arrays; the results for one element
            must be completely independent of any other element. This is not checked. If in doubt,
            use a unary function together with `numpy.vectorize`.
        threshold (Number):
            A divergence threshold to compare the magnitude of the iterated value against.
        max_it (int):
            The maximum number of iterations to apply before assuming sub-threshold convergence.
            Must be positive.
        extent (Tuple[Number, Number, Number, Number]):
            The limits of the x and y axes respectively, in the order (xmin, xmax, ymin, ymax).
            The first element should be less than the second, and the third less than the fourth.
        pixels_per_axis (Number):
            The number of pixels to subdivide each of the x and y axis by. Time taken grows roughly
            quadratically with this value.

    Returns:
        AxesImage:
            The `matplotlib` AxesImage object representing the plot.
    """
    xmin, xmax, ymin, ymax = extent

    x_grid, y_grid = np.meshgrid(
        np.linspace(xmin, xmax, num=pixels_per_axis),
        np.linspace(ymin, ymax, num=pixels_per_axis)
        )
    grid = x_grid + y_grid * 1j

    return ax.imshow(
        max_it+1 - recursion_count(grid, func, threshold=threshold, max_it=max_it),
        extent=extent,
        vmin = 1,
        vmax = max_it+1,
        **imshow_kwargs
        )
