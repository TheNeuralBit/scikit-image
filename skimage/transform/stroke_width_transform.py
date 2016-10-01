import numpy as np
from ..feature import canny
import scipy.ndimage as ndi

from math import atan2, pi, sqrt
from fractions import Fraction

__all__ = ['swt']

def swt(img, dark_on_light=True):
    """Perform a Stroke Width Transform.

    Parameters
    ----------
    img: 2D Array (image)
        Greyscale input image to perform transform on
    dark_on_light: boolean
        Measure stroke widths for dark letters on a light background (True) or
        light letters on a dark background (False)

    Returns
    -------
    output: 2D Array (image)
        Stroke width image

    Notes
    -----

    Examples
    --------

    References
    ----------
    .. [1]
    """
    image_gray = img
    image_edge = canny(image_gray)
    negate_gradient = dark_on_light
    xsobel = ndi.sobel(image_gray, axis=1)
    ysobel = ndi.sobel(image_gray, axis=0)

    widths = np.full(image_edge.shape, np.iinfo(np.uint8).max, np.uint8)
    output = np.full(image_edge.shape, np.iinfo(np.uint8).max, np.uint8)
    rays = []

    # iterate through every edge pixel and look for an opposing edge
    for y0, x0 in _iter_ones(image_edge):
        result = _search_for_opposing_edge(image_edge, xsobel, ysobel, x0, y0, negate_gradient)
        if result:
            x1, y1 = result
            dist = _distance(x0, y0, x1, y1)
            for x, y in _line(x0, y0, x1, y1):
                widths[y][x] = min(widths[y][x], dist)
            rays.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    for r in rays:
       ray_values = [widths[y][x] for x, y in _line(r['x0'], r['y0'], r['x1'], r['y1'])]
       median = np.median(np.array(ray_values))
       for x, y in _line(r['x0'], r['y0'], r['x1'], r['y1']):
           output[y][x] = median

    return output


def _iter_ones(image):
    return zip(*np.where(image))

def _gradient_at(xsobel, ysobel, y, x, negate_gradient):
    horiz = xsobel[y][x]
    vert = ysobel[y][x]

    return (horiz, vert) if not negate_gradient else (-horiz, -vert)

def _search_for_opposing_edge(image_edge, xsobel, ysobel, x, y, negate_gradient):
    dx, dy = _gradient_at(xsobel, ysobel, y, x, negate_gradient)

    def in_bounds(x, y):
        return x >= 0 and y >= 0 and y < image_edge.shape[0] and x < image_edge.shape[1]

    for x1, y1 in _walk_in_direction(dx, dy, pos=(x, y)):
        if not in_bounds(x1, y1): return
        if image_edge[y1][x1]:
            # we've found another edge! check if gradient is in the opposite direction
            dx1, dy1 = _gradient_at(xsobel, ysobel, y1, x1, negate_gradient)
            if pi - pi/6 < abs(atan2(dy1, dx1) - atan2(dy, dx)) < pi + pi/6:
                return (x1, y1)
            else:
                return

def _distance(x0, y0, x1, y1):
    return sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)

# TODO: Credit Rosetta Code
def _line(x0, y0, x1, y1):
    rev = reversed
    if abs(y1 - y0) <= abs(x1 - x0):
        x0, y0, x1, y1 = y0, x0, y1, x1
    else:
        rev = lambda x: x
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    leny = abs(y1 - y0)
    for i in range(leny + 1):
        yield rev((round(Fraction(i, leny) * (x1 - x0)) + x0, (1 if y1 > y0 else -1) * i + y0))

# TODO: Credit Rosetta Code
def _walk_in_direction(dx, dy, pos=(0,0)):
    x, y = pos
    sx = -1 if dx < 0 else 1
    sy = -1 if dy < 0 else 1
    dx = abs(dx)
    dy = abs(dy)
    if dx > dy:
        err = derr = 0 if dy == 0 else dy/dx
        x += sx
        while True:
            yield (x, y)
            err += derr
            if err > 1:
                y += sy
                err -= abs(sy)
            x += sx

    else:
        err = derr = 0 if dx == 0 else abs(dx/dy)
        y += sy
        while True:
            yield (x, y)
            err += derr
            if err > 1:
                x += sx
                err -= abs(sx)
            y += sy




