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

# import of optimizer classes
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
    '''solve quadratic, unconstraint problem with scipy
    returns optimal function value and optimal coordinates'''
    def f(x):
        return  np.dot(x, G).dot(x) + np.dot(a, x)

    constraints = []
        
    result = scipy.optimize.minimize(
        f, x0 = np.zeros(len(G)), method='SLSQP', constraints=constraints,
        tol=1e-10, options={'maxiter': 2000})
    
    return result.fun, result.x
    
def get_quadraticForm(n):
    ''' creates positive semidefinite quadratic (n,n)-matrix, 
    random (n,1) matrix and random (n,1) "+ 1" initial point 
    '''
    # fixed seed for reproducibility 
    # can be comment out if you want to plot different examples for the same dimension
    seed = 5
    np.random.seed(seed)
    
    c = np.matrix(2 * np.random.rand(n, 1) - 1)
    # building A
    B = np.matrix(np.random.rand(n,n))
    Q, _ = np.linalg.qr(B)
    D = np.matrix(np.diagflat(np.random.rand(n)))
    # failsafe 1
    # failsafe non-negative values of D
    res_1 = np.all(np.greater_equal(D, 0))
    if res_1 != True:
        raise ValueError
    A = 10*(Q.T * D * Q)
    # failsafe 2 
    # checking positive semidefiniteness of A
    res = np.all(np.linalg.eigvals(A) > 0)
    if res != True:
        raise ValueError
    # zero as startingpoint
    # xy_init = np.zeros((n,1)) 
    # random 1 + \varepsilon as startingpoint
    xy_init = np.ones((n,1))+ np.random.rand(n, 1)
    return A, c, xy_init

def compare(optimzer_list, dim = 5,  n_iter = 500, tol = 0.01, **optimizer_kwargs):
    '''compares optimzers from optimzerlist for given dimension max number of iterations and tolerance'''
    filename = ""
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00']
    j = 0 # color_helper
    for i in optimzer_list:
        filename+= " " + i +","
        optimizer_class = eval(i)
        # finding optimum and transformation in tensors
        A_, c_, xy_init_ = get_quadraticForm(dim)
        fx_opt, xy_optimal  = solve_qp_scipy(np.asarray(A_), c_.getA1())
        xy_t = torch.tensor(xy_init_, requires_grad=True)
        optimizer = optimizer_class([xy_t], iter = n_iter , **optimizer_kwargs)
        A = torch.tensor(A_)
        c= torch.tensor(c_)
        def f(x):
            return  np.dot(x, np.asarray(A_)).dot(x) + np.dot(c_.getA1(), x)
        # storing results
        inputs=[]
        results =[]
        for t in tqdm(range(1, n_iter + 1)):
            
            optimizer.zero_grad()
            
            loss =quadratic_form(xy_t,A, c)  
            loss.backward()
            # performes optimiziation step
            optimizer.step()
            # quality control
            # averages of \bar{x}_T
            if (i in  ["nFTRL" ,"nOGD" ,"nAda_Grad" , "Ada_Grad"]):
                abs_tol =  np.abs( f(optimizer.average().detach().numpy().flatten()) - fx_opt)
            # quality control
            else: 
                abs_tol = np.abs(f(xy_t.detach().numpy().flatten())- fx_opt)
            inputs.append(t)
            results.append(abs_tol)
            # optimality criterion
            # absulute tolarance and small gradient
            if ( (abs_tol<=tol) or(torch.linalg.norm(xy_t.grad)<=tol)):
                print(i, abs_tol)
                break
        print(i, abs_tol)
        # plot with legends
        if i=="OGD":
            plt.plot(inputs, results, label=i+" "+ r"$\alpha$=1.0", color = colors[j])
        else: 
            # alpha value is needed, otherwise you can't see every plot
            plt.plot(inputs, results, label=i, color = colors[j], alpha =0.7)
            
        j+=1
    # show the plot
    ax = plt.gca()
    plt.title("Comparing "+ filename[:-1] +"\n dim = "+ str(dim))
    #ax.set_xlim([-0.1, 200])
    ax.set_xlabel('# of iterations')
    ax.set_ylabel('Difference\n'r'$ f(\bar{x}_{t}) -f(x^{*})$')
    # just interesting part
    ax.set_ylim([0., 1.])
    plt.legend()
    plt.show()   
    plt.savefig(filename[:-1])
    plt.close()
    
def run_optimization(optimizer_, dim = 2, n_iter = 1500, tol = 0.01, lr = 1., **optimizer_kwargs):
    '''runs optimiazition for one optimizer for given dimension max number of iterations and tolerance'''
    path = np.empty((n_iter + 1, 2))
    
    optimizer_class = eval(optimizer_)
    A_, c_, xy_init_ = get_quadraticForm(dim)
    fx_opt, xy_optimal  = solve_qp_scipy(np.asarray(A_), c_.getA1())
    #print(xy_optimal)
    xy_t = torch.tensor(xy_init_, requires_grad=True)
    A = torch.tensor(A_)
    c= torch.tensor(c_)   
    #print(xy_init_.flatten())           
    path[0, :] = xy_init_.flatten()
    optimizer = optimizer_class([xy_t], iter = n_iter, alpha = lr , **optimizer_kwargs)
    
    inputs=[]
    results =[]
    def f(x):
            return  np.dot(x, np.asarray(A_)).dot(x) + np.dot(c_.getA1(), x)
    for t in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()     
        loss =quadratic_form(xy_t,A, c)
        loss.backward()
                
        optimizer.step()
        # quality control
        # averages of \bar{x}_T
        if (optimizer_ in  ["nFTRL" ,"nOGD" ,"nAda_Grad" , "Ada_Grad"]):
            abs_tol = np.linalg.norm( fx_opt - f(optimizer.average().detach().numpy().flatten()))
           
        # quality control
        else: 
            abs_tol = np.linalg.norm(xy_optimal - f(xy_t.detach().numpy().flatten()))

        inputs.append(t)
        results.append(abs_tol)
         
        if ( (abs_tol<=tol) or(torch.linalg.norm(xy_t.grad)<tol)): # small diff to optimal solution or small diff to grada == 0
           
            break
        path[t, :] = xy_t.detach().numpy().flatten()
 
    plt.plot(inputs, results, label=optimizer_)
    # show the plot
    ax = plt.gca()
    # just interesting part      
    ax.set_ylim([0., 1.])
    plt.legend()
    plt.show()
    return A_, c_, xy_init_, path
    

if __name__ == "__main__":
    n = 2
    temp_input =[n, "OGD",  "nOGD" ,  "nAda_Grad", "Ada_Grad" , "FTRL" , "nFTRL" , "nKT"]
    k = 5
    num_optimzers = len(temp_input)
    # compares normalized with simple version of the algorithm
    #compare(["Ada_Grad", "nAda_Grad"], k)
    #compare(["nOGD", "OGD"], k)
    #compare(["nFTRL", "FTRL"], k)
    #compare(["nKT"], k)
    
    # compares all
    #compare(["OGD", "nOGD","FTRL", "nFTRL","Ada_Grad", "nAda_Grad", "nKT"], k)
    # create animation
    #A_, c_, xy_, path_KT = run_optimization("nKT")
    
    #_, _, _ ,path_nOGD = run_optimization("nOGD")
    #_, _, _ ,path_OGD = run_optimization("OGD")
    #_, _, _ ,path_nFTRL = run_optimization("nFTRL")
    #_, _, _ ,path_FTRL = run_optimization("FTRL")
    #_, _, _ ,path_Ada = run_optimization("Ada_Grad")
    #_, _, _ ,path_nAda = run_optimization("nAda_Grad")
    
