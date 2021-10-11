# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:51:21 2021

@author: Jukka Isohatala
"""

from abc import ABC, abstractmethod

import numpy as np


class RiskFunction(ABC):
    """Base class for risk functions. It is assumed that the risk function
    depends on some basic input parameters x (e.g. the distribution of a 
    random variable on a discrete grid) and possibly additionally
    some free parameters u. The free parameters, if present, u are assumed to 
    be a part of the definition of the risk function: If x -> R is the risk
    function, we suppose 
    
      R(x) = inf_u R0(x, u) for all x.
      
    The purpose of this class is to provide means for evaluating R0
    along with its gradient and Hessian -- the minimization is to be
    computed outside of this class, in the problem where the risk
    function appears as the objective. Optimization of u -> R0 is then
    carried out by extending the control space to include u.

    
    """         
    
    
    @abstractmethod
    def eval(self, x, u=None, new_input=True):
        """Evaluate the risk function at data x and 
        parameters u. If new_imput is false, then one of the eval_* functions
        has been called with the same values x, u without
        a change in them in between, and a hence a cached result
        may be used.
        
        """
        
        pass

    
    @abstractmethod
    def eval_grad(self, x, u=None, new_input=True):
        """Evaluate the gradient of the risk function.
        The function is assumed to be a function of (x, u),
        and the returned gradient should be (G_x, G_u), 
        where G_x and G_u are the gradients w.r.t. x and u, respectively.
        See eval for usage of new_input.
        
        """
        
        pass

    
    @abstractmethod
    def eval_hess(self, x, u=None, new_input=True):
        """Evaluate the Hessian of the risk function. The returned
        matrix should have the shape N x N, where N is the 
        sum of the length of x and u. Only the lower triangle
        will be assumed to be referenced. The returned matrix
        should contained the second derivatives w.r.t. x and u both, that is,
        
          H = [[ H_xx, H_xu ],
               [ H_ux, H_uu ]].
        
        See eval for usage of new_inputs.
        """

        pass

    
    def get_n_infpars(self):
        """Return the length of the u parameter array. """        
        
        return 0

    

class SDTypeRiskFunction(RiskFunction):
    """Class of risk functions based on smoothed positive part functions."""    
    
    
    def __init__(self, eps):        
        self.eps = eps
        
    def spos(self, x):
        """Evaluate the smoothed positive part function at x."""
        
        if x < -36*self.eps: return 0.0
        if x > +36*self.eps: return x
        
        return x + self.eps*np.log(1. + np.exp(-x/self.eps))

    
    def U(self, x):
        """Evaluate the derivative of the smoothed positive part function,
        that is, the smoothed unit step or sigmoid function.
        
        """
    
        if x < -36*self.eps: return 0.0
        if x > +36*self.eps: return 1.0   
        
        z = np.exp(-x/self.eps)
        
        return 1. / (1. + z)

    
    def U_prime(self, x):
        """Evaluate the derivative of the smoothed unit step function.
        """
        
        if abs(x) > 36*self.eps: return 0.0
        
        z = np.exp(-x/self.eps)
        
        return z / (self.eps * (1. + z)**2)        

        
    
class GridSmoothedSemideviation(SDTypeRiskFunction):
    """Smoothed semideviation evaluated on probability measures
    on discrete grid point values.
    
    """
    
    def __init__(self, eps, beta, coords):
        super(GridSmoothedSemideviation, self).__init__(eps)
        
        n = len(coords)
        
        self.beta = beta
        self.ys = coords
        self._val = 0.0
        self._grad = np.zeros(n)
        self._hess = np.zeros((n, n))
        
        
    def _eval_all(self, mu):
        """Evaluate the risk function value, gradient, and hessian
        for input mu. Used internally to update all of these
        when a new input vector is given. It is assumed that typically,
        if the value of the risk function is needed, then so are the
        gradient and the Hessian. This may not be desirable behaviour,
        and should be benchmarked?
        
        """
        
        n = len(mu)
        if n != len(self.ys): 
            raise ValueError(("The input vector length {} does not match the "
                              + "size of the coordinate array given at "
                              + "initialization, {}").format(n, len(self.ys)))
              
        m = np.dot(self.ys, mu)
        
        s = sum(self.spos(self.ys[i] - m) * mu[i]  for i in range(n))      
        
        self._val = m + self.beta * s
        
        for i in range(n):
            s = sum(self.U(self.ys[j] - m)*mu[j] for j in range(n))
            self._grad[i] = self.ys[i] \
                + self.beta*(self.spos(self.ys[i] - m) 
                             - self.ys[i] * s)
            
        for i in range(n):
            for j in range(n):
                s = sum(self.U_prime(self.ys[k] - m) * mu[k] 
                        for k in range(n))
                self._hess[i, j] = self.beta * (self.ys[i]*self.ys[j]*s 
                                     - self.U(self.ys[i] - m)*self.ys[j] 
                                     - self.U(self.ys[j] - m)*self.ys[i])


    def eval(self, mu, u=None, new_input=True):
        
        if new_input: self._eval_all(mu)
                   
        return self._val


    def eval_grad(self, mu, u=None, new_input=True):
                    
        if new_input: self._eval_all(mu)
                
        return self._grad


    def eval_hess(self, mu, u=None, new_input=True):

        if new_input: self._eval_all(mu)
        
        return self._hess
    
        
