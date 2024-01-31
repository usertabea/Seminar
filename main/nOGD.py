# NOGD
import numpy as np
import torch

from torch.optim.optimizer import Optimizer

class nOGD(Optimizer):
    r"""Implements normalized Online Gradient Descent Algorithm 
    version of paper and constant learning rate alpha/sqrt(T)
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
        self._alpha = alpha/np.sqrt(iter)  # constant learing rate alpha/sqrt(T)
        self._iter = iter
        self._step = 0
        self._eps = 1e-8
        self._grad_norm = 0
        # beginning true t=0 
        self._firstep = True
        self._iter= iter
        super(nOGD, self).__init__(params, defaults)
    
    def average(self):
        ''' Returns the weighted average of the iterations before'''
        for group in self.param_groups:
            sum_gradients = group['sum_gradients']
            for p, sum in zip(group['params'], sum_gradients):
                # weighted average
                if self._grad_norm!=0:
                    # x_T = 1/(sum (1/norm(g_t))) * sum x_t/\norm(g_t) 
                    return torch.mul( sum,  1/self._grad_norm)
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
                # x_0 t_0...
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
                    if torch.linalg.norm(grad).item() > 0:
                        # update sum of the 1/norms of the gradients
                        self._grad_norm += 1 /(torch.linalg.norm(grad).item() )
                        # g_t/norm(g_t) normalized gradient
                        g_hat = (grad/(torch.linalg.norm(grad).item() ))
                        # x_t+1 = x_t -alpha/sqrt(T) * g_t/norm(g_t)
                        p.data.copy_(torch.add(p.data, g_hat, alpha = - self._alpha))
                        # udpate sum of x_t / norm(g_t)
                        sum.add_(p.data/(torch.linalg.norm(grad).item()))
                   
        return loss