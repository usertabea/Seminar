# Seminar
## Seminar Project

Implementation of different algorithms and their normaliziation counterparts in the pytorch framework.
The file on_test.py implements methods to run an optimiziation, compare different optimizers and hopefully shows an nice example.

# Implemtation of Algorithm

This project implements three different Algorithms and their normalized version. 
The Implemenation is done using pyorch. Everey optimizer has is own file using the framework of the pytorch-OptimizerClass.

# Test File
The methods in on_test are used as
```
run_optimiziation(optimizer, dim, max_iterations, tolerance)
```
to run the optimiziation with the specified optimizer, plots using pyplot and returns an path containing the iterates the optimiziation method generates.
''' 
compare(optimzer_1, ..., optimizer_n, dim, max_iterations, tolerance)
'''
to compare n different optimizers and show the convergence rates in pyplot. If no dim, number of iteration or tolerance is given, the method uses the default ones.

# Usage
```python
import on_test

# compares FTRL and normalized FTRL on a 5-dim random convex quadratic programming 
# with a maximum number of iterations of 500 and a tolerance of 0.01
on_test.compare(["FTRL", "nFTRL"] , dim = 5,  n_iter = 500, tol = 0.01)
```




