from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam, SGD
from tqdm import tqdm
import math
# working OGD, DG, FTRL
# convernace find in rosen, camel...
#from FTRL_lin import FTRL
from norm_grad import norm_Grad
from norm_OGD import norm_OGD
from KT import KT
from pf import parameterfree
from FTLR_prob import FTRL
# Test function import
from  Test_Functions import *

def run_optimization(xy_init, function_test, optimizer_class,  n_iter, **optimizer_kwargs):
    """Run optimization finding the minimum of the Rosenbrock function.

    Parameters
    ----------
    xy_init : tuple
        Two floats representing the x resp. y coordinates.
    
    function_test : name of function class
    
    optimizer_class : object
        Optimizer class.

    n_iter : int
        Number of iterations to run the optimization for.

    optimizer_kwargs : dict
        Additional parameters to be passed into the optimizer.

    Returns
    -------
    path : np.ndarray
        2D array of shape `(n_iter + 1, 2)`. Where the rows represent the
        iteration and the columns represent the x resp. y coordinates.
    """
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = function_test(xy_t)
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        optimizer.step()
        
        path[i, :] = xy_t.detach().numpy()

    return path
def create_animation(paths,
                     colors,
                     function_test,
                     names,
                     figsize=(12, 12),
                     x_lim=(-2, 2),
                     y_lim=(-1, 3),
                     n_seconds=5):
    """Create an animation.

    Parameters
    ----------
    paths : list
        List of arrays representing the paths (history of x,y coordinates) the
        optimizer went through.

    colors :  list
        List of strings representing colors for each path.

    function_test : name of function class
    
    names : list
        List of strings representing names for each path.

    figsize : tuple
        Size of the figure.

    x_lim, y_lim : tuple
        Range of the x resp. y axis.

    n_seconds : int
        Number of seconds the animation should last.

    Returns
    -------
    anim : FuncAnimation
        Animation of the paths of all the optimizers.
    """
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
   
    Z = function_test([X,Y])
    minimum = (1., 1.)
    start = (0.3, 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")
    ax.plot(*start, "ro")
    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

    return anim


if __name__ == "__main__":
    
    function_test = rosenbrock
    xy_init = (.3, .8)
    n_iter = 1500
    #n_iter = 20
    path_adam = run_optimization(xy_init, function_test, Adam, n_iter)
    path_sgd = run_optimization(xy_init, function_test, SGD, n_iter, lr=1e-3)
    path_weird = run_optimization(xy_init, function_test, FTRL, n_iter)
    path_weird_2 = run_optimization(xy_init,function_test,  norm_Grad, n_iter)
    # OGD
    path_weird_3 = run_optimization(xy_init, function_test,norm_OGD, n_iter , alpha = 1/np.sqrt(n_iter))
    path_weird_pf = run_optimization(xy_init,function_test, parameterfree, n_iter )
    
    freq = 10

    paths = [path_weird_2[::freq], path_sgd[::freq], path_weird[::freq], path_weird_3[::freq], path_adam[::freq]]
    colors = ["green", "blue", "black", "red", "orange"]
    names = ["nGD", "SGD", "FTRL", "OGD", "ADAM"]

    anim = create_animation(paths,
                            colors,
                            function_test,
                            names,
                            figsize=(12, 7),
                            x_lim=(-.1, 1.1),
                            y_lim=(-.1, 1.1),
                            n_seconds=7)

    anim.save("result.gif")
    print(path_sgd[-15:])
    print(path_weird_3[-15:])
    print("WWWW")
    print(path_weird_2[-15:])
    print(path_weird[-15:])
    print(path_weird_pf[-15:])
    #print(path_weird_kt[-15:])