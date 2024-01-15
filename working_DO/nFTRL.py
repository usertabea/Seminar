# test 11.1. FTRL
# example 7.11 / update from paper  linearized loss 
# working for ML
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class nFTRL(Optimizer):
    r"""Implements normalized FTRL algorithm with linearized losses
    version of ML 2019 Orabano Chapter 7 
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
        
        defaults = dict(alpha=alpha)
        # static paramters
        self._alpha = alpha
        self._iter = iter
        self._step = 0
        self._eps = 1e-8
        self._grad_norm = 0
        # beginning true t=0 
        self._firstep = True
        
        super(nFTRL, self).__init__(params, defaults)

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
                # x_0 t_0..
                sum_gradients = group['sum_gradients'] = [torch.zeros_like(p).detach() for p in group['params']]
                x0 = group['x0'] = [torch.clone(p).detach() for p in group['params']]                  
                self._firstep = False
                
            else:
                sum_gradients = group['sum_gradients'] 
                x0 = group['x0']
                
            self._step+=1
            for p, sum, x in zip(group['params'], sum_gradients, x0):
                if p.grad is None:
                    continue
                else:
                    grad = p.grad
                    if torch.linalg.norm(grad).item()> 0:
                        # update sum of the 1/norms of the gradients
                        self._grad_norm += 1 /(torch.linalg.norm(grad).item() )
                        # g_t/norm(g_t)
                        g_hat = (grad/(torch.linalg.norm(grad).item() ))
                        # sum of normed gradients
                        sum.add_(g_hat)
                        # x_t+1 =  - alpha/sqrt(t) * sum_i g_i/norm(g_i)
                        p.data.copy_(torch.mul(sum, - self._alpha/ (np.sqrt(self._step) )))
                        # projection step
                        if (torch.linalg.norm(grad).item())>0:   
                           p.data.copy_(torch.mul(p.data, min(1, 1/(torch.linalg.norm(grad).item()))))
                        #if (torch.linalg.norm(p.data).item())>0:
                        #    p.data.copy_(torch.mul(p.data, min(1, 1/(torch.linalg.norm(p.data).item()))))  
                        
        return loss
