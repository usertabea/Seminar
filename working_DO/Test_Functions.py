from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam, SGD
from tqdm import tqdm
import math
# test function module
# 2-dim functions for optimization
# xy as coordinates, dann trennen! Damit pytorch tensor can handle it, not with np biult ins.
def rosenbrock(xy):
    """Evaluate Rosenbrock function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Rosenbrock function evaluated at the point `xy`.
    """
    x, y = xy

    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def camel_func(xy):
    """Evaluate camelfunction with 3 bumps function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Rosenbrock function evaluated at the point `xy`.
    """
    x, y = xy
    return 2*x**2 -1.05*x**2 +(x**6)/6+ x*y +y**2

def func_easy(xy):
    """Evaluate parabola function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Rosenbrock function evaluated at the point `xy`.
    """
    x, y = xy

    return x ** 2 + y ** 2 

def func_booth(xy):
    """Evaluate booth function function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Rosenbrock function evaluated at the point `xy`.
    """
    x, y = xy

    return (x+2*y-7) ** 2 + (2*x +y -5)** 2 

def func_himmelblau(xy):
    """Evaluate himmelblau function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Rosenbrock function evaluated at the point `xy`.
    """
    x, y = xy

    return (x+2*y-11) ** 2 + (2*x +y -7)** 2 
