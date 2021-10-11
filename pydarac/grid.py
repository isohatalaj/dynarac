# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:04:01 2021

@author: Jukka Isohatala
"""

import math
import itertools as it

import numpy as np

from abc import ABC, abstractmethod


class Grid(ABC):
    
    @abstractmethod
    def indices(self): 
        pass
    
    @abstractmethod
    def get_coords(self, i):
        pass
    
    @abstractmethod
    def get_red_index(self):
        pass

    @abstractmethod
    def get_n_tot(self):
        pass


class ProductGrid(Grid):
    
    def __init__(self, *grids):
        
        # TODO: Add options for grid point position calculations.
        # Now always uses interval midpoints, but this can be problematic.
        
        self.dim = len(grids)
        self.grids = grids
        self.ns = tuple(grid.get_n_tot() for grid in grids)
        self.n_tot = math.prod(self.ns)
        
        
    def get_n_tot(self):
        
        return self.n_tot
    
    
    def indices(self):
        
        return it.product(*(grid.indices() for grid in self.grids))
    
    
    def get_coords(self, i_prod):
        
        return tuple(grid.get_coords(i) for i, grid in zip(i_prod, self.grids))
    
    
    def get_red_index(self, i_prod):
        
        if len(i_prod) != self.dim:
            raise ValueError("dimensions of the grid do not match that of "
                             + "the index")
            
        i = self.grids[-1].get_red_index(i_prod[-1])
        
        for d in range(self.dim - 2, -1, -1):
            i *= self.ns[d]
            i += self.grids[d].get_red_index(i_prod[d])

        return i



class RegularGrid:
    
    def __init__(self, ns, lbs, ubs, geoms=None):
        
        self.dim = len(ns)
        self.ns = ns
        self.lbs = lbs
        self.ubs = ubs
        self.geoms = self.dim * ('bound',) if geoms is None else geoms
        
        if self.dim != len(lbs) or self.dim != len(ubs):
            raise ValueError("Length of n, x_lbs, and x_ubs arrays "
                             + "should all be the same")
            
        self.dxs = np.array([(ubs[d] - lbs[d]) / ns[d] 
                             for d in range(self.dim)])


    def get_red_index(self, i_mult):

        dim = len(i_mult)
        assert(dim <= self.dim)

        i = i_mult[dim - 1]
        
        for d in range(1, dim):
            i *= self.ns[dim - d - 1]
            i += i_mult[dim - d - 1]

        return i


    def shift(self, i_mult, d, v):
 
        id_prime = i_mult[d] + v
        
        if self.geoms[d] == 'bound':
            
            if id_prime < 0 or id_prime > self.ns[d] - 1:
                return None
            
        elif self.geoms[d] == 'cylinder':
            
            while id_prime < 0: id_prime += self.ns[d]           
            while id_prime > self.ns[d] - 1: id_prime -= self.ns[d]
        
        else:
            raise ValueError("Unrecognized geometry '{}' along axis {}"
                             .format(self.geoms[d], d))
            
        return i_mult[:d] + (id_prime,) + i_mult[d+1:]


    def get_n_tot(self):
        
        return math.prod(self.ns)


    def get_coords(self, i_mult):
        
        return np.array([(i_mult[d] + 0.5)*self.dxs[d] + self.lbs[d] 
                         for d in range(self.dim)])
        

    def indices(self):
        
        return it.product(*[range(n) for n in self.ns])

    
    def get_index_from_coord(self, x):
        
        return tuple(min(self.ns[i] - 1, max(0, int((x[i] - self.lbs[i]) 
                                                 / self.dxs[i])))
                     for i in range(self.dim))
    


def main():
    
    g_1 = RegularGrid([3, 3], [-1, -2], [0, 1])
    g_2 = RegularGrid([10], [0], [5])
    
    g_p = ProductGrid(g_1, g_2)
    
    for p_index in g_p.indices():
        
        r_index = g_p.get_red_index(p_index)
        coord = g_p.get_coords(p_index)
        
        q_index = g_2.get_index_from_coord(coord[1])        
        
        print(p_index, r_index, coord, q_index)
        
        
if __name__ == "__main__":
    main()
    
        
        
    
        
    