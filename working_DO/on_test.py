#!/usr/bin/python

from matplotlib.animation import FuncAnimation
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import getopt, sys
import scipy.optimize
import scipy.stats

# import of optimzer classes
from nKT import nKT
from nAda_Grad import nAda_Grad
from nOGD import nOGD
from nFTRL import nFTRL
from OGD import OGD
from Ada_Grad import Ada_Grad
from lFTRL import FTRL
# Test function import
from  Test_Functions import *


        
def solve_qp_scipy(G, a):
    # solve quadratic, unconstraint problem with scipy
    # return optimal function value and optimal coordinates
    def f(x):
        return  np.dot(x, G).dot(x) + np.dot(a, x)

    constraints = []
        
    result = scipy.optimize.minimize(
        f, x0 = np.zeros(len(G)), method='SLSQP', constraints=constraints,
        tol=1e-10, options={'maxiter': 2000})
   
    return result.fun, result.x
    
def get_quadraticForm(n):
    # creates positive semidefinite quadratic (n,n)-matrix, 
    # random (n,1) matrix and random (n,1) "+ 1" initial point 
    
    # fixed seed for reproducibility
    seed = 5
    np.random.seed(seed)
    c = np.matrix(2 * np.random.rand(n, 1) - 1)
    
    B = np.matrix(np.random.rand(n,n))
    Q, _ = np.linalg.qr(B)
    # U = np.matrix(ortho_group.rvs(dim=n))
    D = np.matrix(np.diagflat(np.random.rand(n)))
    A = Q.T * D * Q
    
    xy_init = np.ones((n,1))+ np.random.rand(n, 1)
    return A, c, xy_init

def compare(optimzer_list, dim = 10,  n_iter = 500, tol = 0.01, **optimizer_kwargs):
    # compares optimzers from optimzerlist
    filename = "compare_"
    for i in optimzer_list:
        filename+= "_" + i
        optimizer_class = eval(i)
            
        A_, c_, xy_init_ = get_quadraticForm(dim)
        _, xy_optimal  = solve_qp_scipy(np.asarray(A_), c_.getA1())
              
        xy_t = torch.tensor(xy_init_, requires_grad=True)
        optimizer = optimizer_class([xy_t], iter = n_iter , **optimizer_kwargs)
        A = torch.tensor(A_)
        c= torch.tensor(c_)
        inputs=[]
        results =[]
        
        for t in tqdm(range(1, n_iter + 1)):
            optimizer.zero_grad()
                
            loss =quadratic_form(xy_t,A, c)
                
            loss.backward()
            # performes optimiziation step
            optimizer.step()
            # quality control
            rel_tol = np.linalg.norm((xy_optimal - xy_t.detach().numpy().flatten())/xy_optimal)
            abs_tol = np.linalg.norm(xy_optimal - xy_t.detach().numpy().flatten())
            inputs.append(t)
            results.append(rel_tol)
            # optimality criterion
            if ( (abs_tol<=tol) or(torch.linalg.norm(xy_t.grad)<tol)):
                print(i, abs_tol)
                break
        print(i, abs_tol)
        plt.plot(inputs, results, label=i)
    # show the plot
    ax = plt.gca()
    #ax.set_xlim([-0.1, 200])
    ax.set_ylim([0., 1.])
    plt.legend()
    plt.show()   
    plt.savefig(filename)
    
def run_optimization(optimizer_, dim = 2, n_iter = 100, tol = 0.01, lr = 1., **optimizer_kwargs):
    # runs optimiazition for one optimizer
    path = np.empty((n_iter + 1, 2))
    
    optimizer_class = eval(optimizer_)
    A_, c_, xy_init_ = get_quadraticForm(dim)
    _, xy_optimal  = solve_qp_scipy(np.asarray(A_), c_.getA1())
    print(xy_optimal)
    xy_t = torch.tensor(xy_init_, requires_grad=True)
    A = torch.tensor(A_)
    c= torch.tensor(c_)   
    print(xy_init_.flatten())           
    path[0, :] = xy_init_.flatten()
    optimizer = optimizer_class([xy_t], iter = n_iter, alpha = lr , **optimizer_kwargs)
    
    inputs=[]
    results =[]
    
    for t in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()     
        loss =quadratic_form(xy_t,A, c)
        loss.backward()
                
        optimizer.step()
        
        rel_tol = np.linalg.norm((xy_optimal - xy_t.detach().numpy().flatten())/xy_optimal)
        abs_tol = np.linalg.norm(xy_optimal - xy_t.detach().numpy().flatten())
        
        inputs.append(t)
        results.append(rel_tol)
                
                
        if ( (abs_tol<=tol) or(torch.linalg.norm(xy_t.grad)<tol)): # small diff to optimal solution or small diff to grada == 0
           
            print(optimizer_, abs_tol)
           
            break
        path[t, :] = xy_t.detach().numpy().flatten()
    print(optimizer_, abs_tol)   
    plt.plot(inputs, results, label=optimizer_)
    # show the plot
    ax = plt.gca()
    # just interesting part      
    ax.set_ylim([0., 1.])
    plt.legend()
    plt.show()
    return A_, c_, xy_init_, path
    
   
            
def create_animation(paths,
                     colors,
                     names,
                     A,
                     c,
                     figsize=(12, 12),
                     x_lim=(-2, 2),
                     y_lim=(-1, 3),
                     n_seconds=10):
    
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    print([X,Y])
    def f(x):
        return np.dot(np.dot(x.T,A),x) - np.dot(c,x)
    #Z = quadratic_form_matrix([X,Y], A, c)
        
    fig, ax = plt.subplots(figsize=figsize)
    plt.contour(x, y, f(x[0:2,:]))
    #ax.contour(X, Y, Z, 90)
        
    scatters = [ax.scatter(None,
                            None,
                            label=label,
                            c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    #ax.plot(*minimum, "rD", color = "black")
    #ax.plot(*start, "rD", color ="black")
    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

    return anim


if __name__ == "__main__":
    n = 2
    temp_input =[n, "OGD",  "nOGD" ,  "nAda_Grad", "Ada_Grad" , "FTRL" , "nFTRL" , "nKT"]

    num_optimzers = len(temp_input)
    # compares normalized with simple version of the algorithm
    compare(["OGD", "nOGD"])
    compare(["FTRL", "nFTRL"])
    compare(["nKT"])
    compare(["Ada_Grad", "nAda_Grad"])
    
    # compares all
    #compare(["OGD", "nOGD","FTRL", "nFTRL","Ada_Grad", "nAda_Grad"])
    
    #A_, c_, xy_, path_KT = run_optimization("nKT")
    
    #_, _, _ ,path_nOGD = run_optimization("nOGD")
    #_, _, _ ,path_OGD = run_optimization("OGD")
    #_, _, _ ,path_nFTRL = run_optimization("nFTRL")
    #_, _, _ ,path_FTRL = run_optimization("FTRL")
    #_, _, _ ,path_Ada = run_optimization("Ada_Grad")
    #_, _, _ ,path_nAda = run_optimization("nAda_Grad")
    #freq = 1

    #paths = [path_nOGD[::freq], path_KT[::freq],path_OGD[::freq], path_nFTRL[::freq], path_FTRL[::freq], path_nAda[::freq], path_Ada[::freq]]
    
    #colors = ["green", "royalblue", "black", "lightcoral", "purple", "red", "pink"]
    #names = ["nOGD", "nKT", "OGD", "nFTRL", "FTRL", "nAda", "Ada"] 
    #anim = create_animation(paths,
    #                        colors,
    #                        names,
    #                        A_,
    #                        c_,
    #                        figsize=(12, 7),
    #                        #x_lim=(-1., 2.),
    #                        #y_lim=(-1., 2.1),
    #                        n_seconds=10)

    #anim.save("result.gif")
    
    