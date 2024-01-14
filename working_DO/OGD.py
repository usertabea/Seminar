# test 28.11.  online gradient descent
# working for ML
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

# mary poppins
class OGD(Optimizer):
    r"""Implements Online Gradient Descent Algorithm with constant learning rate alpha/sqrt(T)
    version of ML Book Chapter 2.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter (default: 1.0)
        iter (int, recommended): maximal number of iterations. (default: 100)
    
    """

    def __init__(self, params, alpha: float = 1., iter : int = 100):
        
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0 < iter:
            raise ValueError("Invalid number of iterations: {}".format(iter))
        
        # alpha is learning rate
        defaults = dict(alpha=alpha)
        # static paramters
        self._alpha = alpha/np.sqrt(iter) # constant learing rate alpha/sqrt(T)
        self._iter = iter
        self._step = 0
        self._eps = 1e-8
        self._grad_norm = 0
        
        # beginning true t=0 
        self._firstep = True
        super(OGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self._firstep:
                #x_0 t_0...
                
                # Sum of the gradients
                sum_gradients = group['sum_gradients'] = [torch.zeros_like(p).detach() for p in group['params']]
                x0 = group['x0'] = [torch.clone(p).detach() for p in group['params']]                  
                self._firstep = False
            else:
                sum_gradients = group['sum_gradients']
                x0 = group['x0']
            
            self._step += 1
            for p, sum in zip(group['params'], sum_gradients):
                if p.grad is None:
                    continue
                else:
                    grad = p.grad
                    if (torch.linalg.norm(grad).item()!=0):
                        
                        # udpate sum of x_t 
                        sum.add_(p.data)
                        # x_t+1 = x_t -alpha/sqrt(T) * g_t
                        p.data.copy_(torch.add(p.data, grad, alpha =  -self._alpha))
                    # last step for Online/Batch
                if  self._step == self._iter +1:
                    # x_T = 1/T * sum x_t 
                    p.data.copy_(torch.mul( sum,  1/self._step))
        return loss