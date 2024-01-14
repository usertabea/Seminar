
import numpy as np
import torch

# test function module
# 2-dim functions for optimization
# evluate xy as coordinates, pytorch tensor can handle it, not with np biult ins.
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

def quadratic_form(xy, A, c):
    """Evaluate Quadratic Form of x^T A x + c^T x.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.
    A : quadratic matrix of n,n dimension.
    c : matrix of n,1 dimension.
    Returns
    -------
    float
        The function evaluated at the point `xy`.
    """
    return torch.add(torch.matmul(torch.matmul(torch.transpose(xy,0,1),A),xy),torch.matmul(torch.transpose(c,0,1),xy))
def quadratic_form_matrix(xy, A, c):
    """Evaluate Quadratic Form of x^T A x + c^T x.
    Matrix-Version for plotting
    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.
    A : quadratic matrix of n,n dimension.
    c : matrix of n,1 dimension.
    Returns
    -------
    float
        The function evaluated at the point `xy`.
    """
    x, y = xy
    # extreme ugly
    return x**2 * A[0][0]+x*y*A[0][1] + x*y*A[1][0] + y**2 *A[1][1] + x*c[0][0]+y*c[0][1]
    # return (xy.T.dot(A)@xy  + c.T.dot(xy)).item(0)

def ackley_tens(xy):
    """Evaluate Ackley function for Tensors.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.
    Returns
    -------
    float
        The ackley function evaluated at the point `xy`.
    """
    x, y = xy
    
    sum_sq_term = -20 * torch.exp(-0.2*torch.sqrt(0.5*(x*x+y*y)))
    cos_term = -torch.exp(0.5*(torch.cos(2*np.pi*x)+torch.cos(2*np.pi*y)))
    value = sum_sq_term+cos_term+np.exp(1)+20
    return value
    
def ackley(xy):
    """Evaluate Ackley function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.
    Returns
    -------
    float
        The ackley function evaluated at the point `xy`.
    """
    x,y = xy
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * 
  np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20
def camel_func(xy):
    """Evaluate camelfunction with 3 bumps.
    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The camel function evaluated at the point `xy`.
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
        The function evaluated at the point `xy`.
    """
    x, y = xy

    return x ** 2 + y ** 2 

def func_booth(xy):
    """Evaluate the booth function function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The booth function evaluated at the point `xy`.
    """
    x, y = xy

    return (x+2*y-7) ** 2 + (2*x +y -5)** 2 

def func_himmelblau(xy):
    """Evaluate Himmelblau function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Himmelblau function evaluated at the point `xy`.
    """
    x, y = xy

    return (x+2*y-11) ** 2 + (2*x +y -7)** 2 
def beale_func(xy):
    """Evaluate Beale function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Beale function evaluated at the point `xy`.
    """
    x, y = xy
    return (1.5 -x + x*y)**2 + (2.25 -x +x*y**2)**2+(2.625 -x + x*y**3)**2