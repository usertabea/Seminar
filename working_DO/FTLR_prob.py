# test 28.11. FTRL
# example 7.11 / update from paper  linearized loss 
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
# works 
# mary poppins

class FTRL(Optimizer):
    r"""Implements FTRL algorithm with linearized losses
    version of ML 2019 Orabano Chapter 7 
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        iter (int, recommended): maximal number of iterations. (default: 10)
        alpha (float, optional): alpha parameter (default: 1.0)
        
    """

    def __init__(self, params, alpha: float = 1., iter : int = 10):
        
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        
        # smoothing parameter for adaptive learning rate
        defaults = dict(alpha=alpha)
        # static paramters
        self._alpha = alpha
        self._iter = iter
        # beginning true t=0 
        self._firstep = True
        
        super(FTRL, self).__init__(params, defaults)

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
            for p in group["params"]:
                if p.grad is None:
                    continue
                # g_t
                grad = p.grad.data
                state = self.state[p]
                # beginning t=0
                if self._firstep:
                    #x_0 t_0...
                    state["step"] = 0
                    # Sum of the gradients
                    state['sum_gradients'] = torch.zeros_like(p).detach()
                    # Sum of the norm of the gradients
                    state['grad_norm_sum'] = 0 
                    # We need to save the initial point
                    state['x0'] = torch.clone(p.data).detach()
                    # not fisrt step
                    self._firstep = False
                    
                state['step'] += 1
                sum_gradients, grad_norm_sum,x0 = (
                    state['sum_gradients'],
                    state['grad_norm_sum'],
                    state['x0'],
                )
                #theta = torch.mul(p.data ,(np.sqrt(state["step"]) / self._alpha -  (np.sqrt(state["step"]-1) / self._alpha)))
                if torch.linalg.norm(grad)!=0: # don#t devide through zero
                    # update sum of the normedgradients 
                    grad_norm_sum = grad_norm_sum + (1 /(torch.linalg.norm(grad).item() ))
                    state["grad_norm_sum"] = grad_norm_sum # weird update 
                    # summation of normed  gradients
                    sum_gradients.add_(p.data/(torch.linalg.norm(grad).item() ))
                    # g_t/norm(g_t)
                    normed_grad = (grad/(torch.linalg.norm(grad).item() ))
                    # x_t+1 = x_t - alpha/sqrt(t) * g_t/norm(g_t)
                    p.data = torch.add(p.data, normed_grad, alpha = - self._alpha/ (np.sqrt(state["step"])))
                # last step for normatian   
                if  (state['step'] == self._iter +1):
                    if grad_norm_sum!=0:
                        # x_T = 1/(sum (1/norm(g_t))) * sum x_t/\norm(g_t) 
                        p.data = torch.mul( sum_gradients,  1/grad_norm_sum)       

        return loss