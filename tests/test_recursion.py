"""Test the recursion.py module.
"""
# pylint: disable=import-error
import numpy as np

from recursion import recursion_count


def test_recursion_count_basic():
    """A very simple case of using `recursion_count`"""
    x_0 = [[1, 2],[3, 4]]
    func = lambda x: x**2
    threshold = 100
    max_it = 10
    count = recursion_count(x_0, func, threshold, max_it)

    # 2**2 = 4 -> 4**2 = 16 -> 16**2 = 256 > 100, count = 3
    # 3**2 = 9 -> 9**2 = 81 -> 81**2 = 6561 > 100, count = 3
    # 4**2 = 16 -> 16**2 = 256 > 100, count = 2

    assert np.allclose(count, [[max_it+1, 3],[3, 2]])


def test_recursion_count_complex():
    """A simple case of using `recursion_count` with complex numbers"""
    x_0 = [[0.5, complex(1,0.5)],[complex(0.25, 0.25), complex(1/3, -5)]]
    func = lambda x: x**2 + 0.25
    threshold = 2
    max_it = 6
    count = recursion_count(x_0, func, threshold, max_it)

    # 0.5**2 + 0.25 = 0.75
    #       -> 0.75**2 + 0.25 = 0.5625
    #       -> 0.5625**2 + 0.25 = 0.56640625
    #       -> 0.0.56640625**2 + 0.25 = 0.57081604003
    #       -> etc. This looks like it converging. count = 7

    # (1+0.5j)**2 + 0.25 = 1+j, |1+j| = sqrt(2) ~ 1.41 < 2
    #       -> (1+j)**2 + 0.25 = 0.25+2j, |0.25+2j| = sqrt(4.0625) ~ 2.02 > 2, count = 2

    # (0.25+0.25j)**2 + 0.25 = 0.25+0.125j, |0.25+0.125j| = sqrt(0.078125) ~ 0.280 < 2
    #       -> (0.25+0.125j)**2 + 0.25 = 0.296875+0.0625j,
    #               |0.296875+0.0625j| = sqrt(0.092) ~ 0.303 < 2
    #       -> (0.296875+0.0625j)**2 + 0.25 = 0.334228516+0.037109375j,
    #               |0.334228516+0.037109375j| = sqrt(0.11308580662) ~ 0.336 < 2
    #       -> (0.334228516+0.037109375j)**2 + 0.25 = 0.360331595+0.0248060227j
    #               |0.360331595+0.0248060227j| ~ 0.36118443642 < 2
    #       -> etc. This looks like it is converging. count = 7.

    # (1/3-5j)**2 + 0.25 = (-887/36)-(3/10)j, |(-887/36)-(3/10)j| ~ 24.6 > 2, count = 1

    assert np.allclose(count, [[max_it+1, 2],[max_it+1, 1]])

