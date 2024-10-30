import math

"""
Annealing support for KL divergence

"""
class Annealer():
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=True, disable=False):
        self.total_steps = total_steps
        self.baseline = baseline
        self.current_step = 0.0
        self.shape = shape
        self.cyclical = cyclical
        if disable:
            self.shape = None
            self.baseline = 0.0
    
    def __call__(self, kl_loss):
        return kl_loss * self.beta()

    def beta(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            x = (self.total_steps/2)-self.current_step
            y = 1/(1+math.exp(x))
        elif self.shape == None:
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        out = y*(1-self.baseline) + self.baseline
        return out

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0

    def set_cyclical(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return
