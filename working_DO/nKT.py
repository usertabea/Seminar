# nKT

import torch
from torch.optim.optimizer import Optimizer

class nKT(Optimizer):
    r"""Implements the normalized KT algorithm from 'Normalized Gradients for ALL' by Orabano.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): Initial wealth d_0. Best between 1 and sqrt(T)
    """
    def __init__(self, params, alpha: float = 1., iter: int = 0):
        if not 0.0 <= alpha:
            raise ValueError("Invalid d_0 value: {}".format(alpha))

        self._wealth = alpha # initial d_0
        self._iter= 0
        self._firstep = True
        self._eps = 1e-8
        super(nKT, self).__init__(params)

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
            weight_decay = group['weight_decay']
            if self._firstep:
                x0 = group['x0'] = [torch.clone(p).detach() for p in group['params']]
                theta = group['theta'] = [torch.zeros_like(p).detach() for p in group['params']]
                zeta = group['zeta'] = [torch.zeros_like(p).detach() for p in group['params']]
                self._firstep = False
            else:
                x0 = group['x0']
                theta = group['theta']
                zeta = group['zeta']
           
            self._iter += 1
            # update the sum of the negative gradients and the weights
            for p, t, z, x in zip(group['params'], theta, zeta, x0):
                if p.grad is None:
                    continue
                else:
                    # z = \sum <g_i/norm(g_i), x_i>
                    z.add_(torch.dot(torch.div(p.grad, torch.linalg.norm(p.grad).item() +self._eps).flatten(), p.detach().flatten()))
                    # temp = d_0 - \sum <g_i/norm(g_i), x_i>
                    temp = torch.add(z, self._wealth, alpha = -1) 
                    # t = -\sum g_i/norm(g_i)
                    t.add_(torch.div(p.grad, torch.linalg.norm(p.grad).item() +self._eps), alpha=-1)
                    # x_t = x_1 + (-\sum g_i/norm(g_i))/t (d_0 - \sum <g_i/norm(g_i), x_i>)
                    p.data.copy_(t.mul(temp/self._iter).add(x))                  
        return loss