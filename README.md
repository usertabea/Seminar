# Seminar
## Seminar Project

Implementation of different algorithms and their normalized counterparts in the pytorch framework.
The file on_test.py implements methods to run an optimization, compare different optimizers and hopefully shows a nice example.

# Implementation of the algorithm

This project implements three different algorithms and their normalized version from ["Normalized Gradients for ALL"] (https://arxiv.org/abs/2308.05621) by F. Orabano and ["Online to Offline Conversions, Universality and Adaptive Minibatch Sizes"] (https://arxiv.org/abs/1705.10499) by K.Y. Levy. 
The implementation is done with Pyorch. Each optimizer has its own file that uses the framework of the pytorch-OptimizerClass.

# Test file
The methods in on_test are used as
```
run_optimiziation(optimizer, dim, max_iterations, tolerance)
```
to run the optimization with the specified optimizer, draws with pyplot and returns a path containing the iterations that the optimization method generates.
```
compare([optimizer_1, ..., optimizer_n], dim, max_iterations, tolerance)
```
to compare n different optimizers and display the absulute difference to the optimal point in pyplot. If no dim, no number of iterations or tolerance is specified, the method uses the default values.


# Usage
```python
import on_test

# compares FTRL and normalized FTRL on a 5-dim random convex quadratic programming 
# with a maximum number of iterations of 500 and a tolerance of 0.01
on_test.compare(["FTRL", "nFTRL"] , dim = 5,  n_iter = 500, tol = 0.01)
```




