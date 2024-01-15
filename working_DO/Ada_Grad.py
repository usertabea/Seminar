import numpy as np
import torch
from torch.optim.optimizer import Optimizer

class Ada_Grad(Optimizer):
    r"""Implements Adaptive Gradient Descent Algorithm (AdaGrad) version of paper
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter as learning rate (default: 1.0) 
        iter (int, recommended): maximal number of iterations. (default: 100)
    """

    def __init__(self, params, alpha: float = 1, iter: int =100):
        
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0 < iter:
            raise ValueError("Invalid number of iterations: {}".format(iter))
        
        defaults = dict(alpha=alpha)
        # first steps and learningrate as static parameter
        self._alpha = alpha
        self._firstep = True
        self._step = 1
        self._norm_grad = 0
        self._G = 1e-8
        self._iter= iter
        super(Ada_Grad, self).__init__(params, defaults)

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
                x0 = group['x0'] = [torch.clone(p).detach() for p in group['params']]
                sum_gradients = group['sum_gradients']= [torch.zeros_like(p).detach() for p in group['params']]
                
                self._firstep = False
            else:
                x0 = group['x0']
                sum_gradients = group['sum_gradients']
            self._step+=1
            for p, sum, x in zip(group['params'], sum_gradients, x0):
                if p.grad is None:
                    continue
                else:
                    # update the sum of the negative gradients and the weights
                    grad = p.grad
                    # update maximum range of the gradients
                    # helper for devision
                    # G >= g_t forall t
                    if self._G < torch.linalg.norm(grad).item() :
                            self._G =torch.linalg.norm(grad).item()
                    if torch.linalg.norm(grad).item()!=0:
                        # udpate sum of x_t
                        sum.add_(p.data)
                        # update sum of the norms of the gradients
                        self._norm_grad += torch.linalg.norm(grad).item() **2 
                        # alpha = alpha/ sqrt(G**2 + sum norm(g_t))
                        normalized_alpha = self._alpha / (np.sqrt((self._G**2 +self._norm_grad)))
                        # x_t+1 = x_t - alpha/(sqrt(G**2 + sum norm(g_t)) * sum g_t
                        p.data.copy_(torch.add(p.data, grad, alpha = - normalized_alpha))   
                    #if  self._step == self._iter -1 :
                    #    # weighted average
                    #    if self._grad_norm!=0:
                    #        # x_T = 1/(sum (1/norm(g_t))) * sum x_t/\norm(g_t) 
                    #        p.data.copy_(torch.mul( sum,  1/self._grad_norm))    
        return loss
