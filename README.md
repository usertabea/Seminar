# Seminar
## Seminar Project

Implementation of different algorithms and their normaliziation counterparts in the pytorch framework.
The file on_test.py implements methods to run an optimiziation, compare different optimizers and hopefully shows an nice example.

# Implemtation of Algorithm

This project implements three different Algorithms and their normalized version.
The Implemenation is done using pyorch. Everey optimizer has is on file using the framework of the pytorch-OptimizerClass.

# Test File
The methods in on_test are used as
```
run_optimiziation(optimizer, dim, max_iterations, tolerance)
```
to run the optimiziation with the specified optimizer, plots using pyplot and returns an path containing the iterates the optimiziation method generates.
```
compare(optimzer1, ..., optimizern, dim, max_iterations, tolerance)
```
to compare n different optimizers and show the convergence rates in pyplot. If no dim, number of iteration or tolerance is given, the method uses the default ones.




