# test 29.11. normalized gradient descent

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
#does something right i guess
# shows zig zagging of GD
# updates working, just looking shitty
# furchtbar schnell keine Ahnung ob richtig ;)
# mary poppins

class norm_Grad(Optimizer):
    r"""Implements normalized Gradient Descent Algorithm version of paper
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter as learning rate (default: 1.0) 
    
    """

    def __init__(self, params, alpha: float = 1):
        
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        
        defaults = dict(alpha=alpha)
        # first steps and learningrate as static parameter
        self._alpha = alpha
        self._firstep = True
        super(norm_Grad, self).__init__(params, defaults)

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
                    # Maximum observed scale 
                    # helper for zero devison 
                    state['G'] = 1e-7
                    # We need to save the initial point
                    state['x0'] = torch.clone(p.data).detach()
                    self._firstep = False
                state['step'] += 1
                sum_gradients, grad_norm_sum, G, x0 = (
                    state['sum_gradients'],
                    state['grad_norm_sum'],
                    state['G'],
                    state['x0'],
                )
                # update maximum range of the gradients
                # helper for devision
                # G >= g_t forall t
                if G < torch.linalg.norm(grad).item() :
                    G = torch.linalg.norm(grad).item()
                    state["G"] = G # update
                # udpate sum of gradients
                sum_gradients.add_(grad)
                # update sum of the norms of the gradients
                grad_norm_sum = grad_norm_sum + (torch.linalg.norm(grad).item() **2 )
                state["grad_norm_sum"] = grad_norm_sum # update
                # alpha = alpha/ sqrt(G**2 + sum norm(g_t))
                normalized_alpha= self._alpha / (np.sqrt((G**2 +grad_norm_sum)))
                # x_t+1 = x_0 - alpha/(sqrt(G**2 + sum norm(g_t)) * sum g_t)
                p.data = torch.add(x0, sum_gradients, alpha = - normalized_alpha)
               
        return loss