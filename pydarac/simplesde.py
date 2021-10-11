
import itertools as it
import math
import sys

import numpy as np

from riskawarenlp import *
from grid import *


class RASimpleSDE(RiskAwareControlNLP):
    """Risk-aware control of differential equations with
    a diagonal diffusion matrix on a rectangular grid."""

    def __init__(self, grid_x, grid_y, grid_q, grid_a, grid_t, risk_func,
                 use_log_y=False):
        
        """If use_log_y is True, then internally track the log of running
        costs. This does not affect the output distribution q. It is up
        to the user to setup the y grid in a sensible way.
        """
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_q = grid_q
        self.grid_a = grid_a
        self.grid_t = grid_t
        
        self.grid_xyat = ProductGrid(grid_x, grid_y,
                                     grid_a, grid_t)
        
        self.grid_xyt = ProductGrid(grid_x, grid_y, grid_t)
           
        n_disc_tot_x = grid_x.get_n_tot()
        n_disc_tot_a = grid_a.get_n_tot()
        n_disc_tot_xa = n_disc_tot_x * n_disc_tot_a
        n_disc_y = grid_y.get_n_tot()
        n_disc_q = grid_q.get_n_tot()
        n_disc_t = grid_t.get_n_tot()
        
        self.use_log_y = use_log_y
        
        # For the purposes of calculating the running-cost, terminal
        # cost projection onto the total cost axis q.
        self.k_proj_supsample_fact = 16
        
        super(RASimpleSDE, self).__init__(n_disc_tot_x, n_disc_tot_xa,
                                          n_disc_y, n_disc_q, n_disc_t,
                                          risk_func)


    def c_cost(x, a, t):
        """Evaluate the cost rate function at (x, a, t)."""

        return 0.0
    

    def v_cost(x):
        """Evaluate the terminal cost at x."""

        return 0.0


    def b_drift(i, x, a, t):
        """Return the drift vector component i at (x, a, t)."""

        raise NotImplementedError("b_drift must be implemented")


    def sigma_diffusion(i, x, a, t):
        """Return the diffusion rates at (x, a, t). The model
        used in this class assumes a diagonal diffusion matrix,
        and this function should return the ith diagonal entry 
        in the matrix."""

        raise NotImplementedError("sigma_diffusion must be implemented")


    def sigma_diffusion_prime(i, x, a, t):
        """Return the x_i derivative of sigma_diffusion(i, x, a, t)."""

        raise NotImplementedError("sigma_diffusion_prime must be implemented")

   
    def setup_q_projection(self, P):
        """When calculating the projectoion matrix, the X- and Y-axes
        are sampled at a tighter grid spacing. """
        
        ss_fact = self.k_proj_supsample_fact
        weight = (1.0/ss_fact)**2
        
        # TODO: A grid copy method or something? The grid
        # geometries are ignored here, fine for now, but may
        # cause problems in the future?
        ss_grid_x = RegularGrid([ss_fact*n for n in self.grid_x.ns],
                                self.grid_x.lbs, self.grid_x.ubs)
        ss_grid_y = RegularGrid([ss_fact*n for n in self.grid_y.ns],
                                self.grid_y.lbs, self.grid_y.ubs)

        l = (self.n_disc_t - 1,)
        for (ss_i, ss_j) in it.product(ss_grid_x.indices(),
                                       ss_grid_y.indices()):
                
            x = ss_grid_x.get_coords(ss_i)
            y = ss_grid_y.get_coords(ss_j)
            v = self.v_cost(x)
            
            if self.use_log_y: 
                tot_cost = np.exp(y[0]) + v
            else:
                tot_cost = y[0] + v
                
            i = self.grid_x.get_index_from_coord(x)
            j = self.grid_y.get_index_from_coord(y)
            
            row = self.grid_q.get_index_from_coord(np.array([tot_cost]))[0]
            
            for k in self.grid_a.indices():
                
                col = self.grid_xyat.get_red_index((i, j, k, l))
                
                P[row, col] += weight


    def get_iv(self):
        
        raise NotImplementedError("get_iv must be implemented")
        
    
    def get_ic(self):
        
        x_init_val = self.get_iv()
        xy_init_vec = np.zeros(self.n_disc_tot_x * self.n_disc_y)

        i_init = self.grid_x.get_index_from_coord(x_init_val)
        j_init = (0,) # self.grid_y.get_index_from_coord(np.array([0.0]))
        
        i_red = self.grid_x.get_red_index(i_init)
        j_red = self.grid_y.get_red_index(j_init)
        
        xy_init_vec[i_red + self.n_disc_tot_x*j_red] = 1.0
        
        return xy_init_vec
                    

    def setup_A_b(self, A, B, ic):
        
        inv_dx = 1.0/self.grid_x.dxs
        inv_dt = 1.0/self.grid_t.dxs[0]

        for (i, j, l) in self.grid_xyt.indices():

            if l[0] == 0:
                continue
            
            row = self.grid_xyt.get_red_index((i, j, l))
            
            B[row] = 0.0

            for k in self.grid_a.indices():

                col = self.grid_xyat.get_red_index((i, j, k, l))
                col_l_minus = self.grid_xyat.get_red_index((i, j, k, (l[0] - 1,)))
                col_j_minus = self.grid_xyat.get_red_index((i, (j[0] - 1,), k, l))
                col_j_plus = self.grid_xyat.get_red_index((i, (j[0] + 1,), k, l))
                                                           
                (x, y, a, t) = self.grid_xyat.get_coords((i, j, k, l))
                
                y = y[0]
                t = t[0]

                # timestep
                A[row, col] += -inv_dt
                A[row, col_l_minus] += +inv_dt

                # Handle first the Y-space direction, upwind scheme
                # used here
                if self.grid_y.ns[0] > 1:

                    j_minus = self.grid_y.shift(j, 0, -1)
                    j_plus = self.grid_y.shift(j, 0, +1)
                    
                    # Enforce no flux over boundaries by checking
                    # which flux components we need at this point
                    if j_minus is None:
                        j_deltas = [0]
                    elif j_plus is None:
                        j_deltas = [-1]
                    else:
                        j_deltas = [0, -1] 

                    for j_delta in j_deltas:
                        
                        dy = self.grid_y.dxs[0]
                        
                        j_use = self.grid_y.shift(j, 0, j_delta)
                        j_use_plus = self.grid_y.shift(j, 0, j_delta + 1)
                        
                        y_mid = self.grid_y.get_coords(j_use)
                        y_mid[0] += 0.5 * dy # prev'ly incorrectly w/o 0.5
                        
                        b_mid = self.c_cost(x, a, t)
                        if self.use_log_y:
                            b_mid = np.exp(-y_mid[0])*b_mid
                       
                        if b_mid < 0.0:                            
                            cc_delta = 0.0
                        else:
                            cc_delta = 1.0                            
                            
                        w_j_plus = -b_mid*(1.0 - cc_delta)
                        w_j = -b_mid*cc_delta
                        
                        col = self.grid_xyat.get_red_index((i, j_use, k, l))
                        col_plus = self.grid_xyat.get_red_index((i, j_use_plus, k, l))
                        
                        if j_delta == 0:
                            A[row, col] += w_j / dy
                            A[row, col_plus] += w_j_plus / dy
                        else:
                            A[row, col] += -w_j / dy
                            A[row, col_plus] += -w_j_plus / dy


                # Then X-space, using Chang-Cooper -- NO, Chang-Cooper
                # does not work! Revert to upwind. 
                # TODO: Make discretization selectable
                # TODO (hard): Figure out a discretization methods
                # that supports all relaxed controls.
                # TODO: Optimize this, e.g. every interior flux is evaluated
                # twice, etc.
                for i_dir in range(self.grid_x.dim):

                    i_minus = self.grid_x.shift(i, i_dir, -1)
                    i_plus = self.grid_x.shift(i, i_dir, +1)
                    
                    # Enforce no flux over boundaries by checking
                    # which flux components we need at this point
                    if i_minus is None:
                        i_deltas = [0]
                    elif i_plus is None:
                        i_deltas = [-1]
                    else:
                        i_deltas = [0, -1] 

                    for i_delta in i_deltas:
                        
                        dx = self.grid_x.dxs[i_dir]
                        
                        i_use = self.grid_x.shift(i, i_dir, i_delta)
                        i_use_plus = self.grid_x.shift(i, i_dir, i_delta + 1)
                        
                        x_mid = self.grid_x.get_coords(i_use)
                        x_mid[i_dir] += 0.5 * dx  # prev'ly incorrectly w/o 0.5
                        
                        b_mid = self.b_drift(i_dir, x_mid, a, t)
                        s_mid = self.sigma_diffusion(i_dir, x_mid, a, t)
                        D_mid = 0.5*s_mid**2
                        Dp_mid = s_mid*self.sigma_diffusion_prime(i_dir, x_mid, a, t)
                        
                        cc_lambda = dx*(-b_mid + Dp_mid) / D_mid

                        # TODO: Use an approximation near cc_lambda == 0,
                        # there is no division by zero!                                            
                        cc_delta = 1.0/cc_lambda + 1.0/(1.0 - np.exp(np.clip(cc_lambda,-36,36)))
                        
                        # FORCE MIDPOINT
                        cc_delta = 0.5

                        # FORCE UPWIND
                        if cc_lambda > 0.0:                            
                            cc_delta = 0.0
                        else:
                            cc_delta = 1.0        
                        
                        c_mid = -b_mid + Dp_mid
                        
                        w_i_plus = c_mid*(1 - cc_delta) + D_mid / dx
                        w_i = c_mid*cc_delta - D_mid / dx
                        
                        col = self.grid_xyat.get_red_index((i_use, j, k, l))
                        col_plus = self.grid_xyat.get_red_index((i_use_plus, j, k, l))
                        
                        if i_delta == 0:
                            A[row, col] += w_i / dx
                            A[row, col_plus] += w_i_plus / dx
                        else:
                            A[row, col] += -w_i / dx
                            A[row, col_plus] += -w_i_plus / dx

            # column loop ends
        # row loop ends

        # Initial condition loop
        for (i, j) in it.product(self.grid_x.indices(),
                                 self.grid_y.indices()):

            l = (0,)
            row = self.grid_xyt.get_red_index((i, j, l))
            B[row] = ic[row]
            
            for k in self.grid_a.indices():
                
                col = self.grid_xyat.get_red_index((i, j, k, l))
                
                A[row, col] = 1.0
            
        # Done.

    def intermediate(self,
                     alg_mod,
                     iter_count, obj_value,
                     inf_pr, inf_du,
                     mu,
                     d_norm,
                     regularization_size,
                     alpha_du, alpha_pr,
                     ls_trials):

        pass        


    def print_terminal(self, x, file=sys.stdout, sep=', '):
        
        cost_distr = self.get_cost_distr(x)
        
        for j in self.grid_q.indices():
            
            y = self.grid_q.get_coords(j)
            
            print(" {:4d}".format(j[0]), end=sep, file=file)
            print(" {:25.15f}".format(y[0]), end=sep, file=file)
            print(" {:25.15f}".format(cost_distr[j[0]]), file=file)
            
            
    def print_control(self, sol, file=sys.stdout, sep=', '):
        """Attempt to print out the control process, assuming it is
        strict. The routine prints the indices and the coordinates
        for the value of the control that has the most probability
        mass for each given x, y, t. It also reports the probility
        mass share of the mode of the control -- if always near
        1.0, then the control can reasonably be considered strict.q
        """
        
        for i, j, l in self.grid_xyt.indices():
            
            s = 0.0
            m_max = 0.0
            first = True
            
            for k in self.grid_a.indices():
                
                if first:
                    k_max = k
                    first = False
                
                i_red = self.grid_xyat.get_red_index((i, j, k, l))
                
                m = sol[i_red]
                
                if m > m_max:
                    s += m_max
                    m_max = m
                    k_max = k
                else:
                    s += m
                    
            (x, y, a, t) = self.grid_xyat.get_coords((i, j, k_max, l))
            
            max_share = m_max / (s + m_max)
            
            for ix in i:
                print(" {:4d}".format(ix), end=sep, file=file)

            for jx in j:
                print(" {:4d}".format(jx), end=sep, file=file)

            for kx in k_max:
                print(" {:4d}".format(kx), end=sep, file=file)

            for lx in l:
                print(" {:4d}".format(lx), end=sep, file=file)
                
            for xx in x:
                print(" {:25.15f}".format(xx), end=sep, file=file)

            for yx in y:
                print(" {:25.15f}".format(yx), end=sep, file=file)
                
            for ax in a:
                print(" {:25.15f}".format(ax), end=sep, file=file)

            for tx in t:
                print(" {:25.15f}".format(tx), end=sep, file=file)
                
            print(" {:25.15f}".format(max_share), file=file)
                
                
        
        
    def print_margts(self, x, axes,
                     file=sys.stdout, sep=', '):

        apr = []  # Axes to print
        asm = []  # Axes to sum over
        
        grid_all = RegularGrid(self.grid_x.ns + self.grid_y.ns 
                               + self.grid_a.ns + self.grid_t.ns,
                               self.grid_x.lbs + self.grid_y.lbs 
                               + self.grid_a.lbs + self.grid_t.lbs,
                               self.grid_x.ubs + self.grid_y.ubs 
                               + self.grid_a.ubs + self.grid_t.ubs)

        for a in range(grid_all.dim):
            if a in axes: asm = asm + [a]
            else: apr = apr + [a]
        
        grid_marg = RegularGrid([grid_all.ns[d] for d in apr],
                                [grid_all.lbs[d] for d in apr],
                                [grid_all.ubs[d] for d in apr])
        
        x_marg = np.zeros(math.prod(grid_marg.ns))

        
        for i in grid_all.indices():
            
            i_marg = tuple(i[d] for d in apr)
            
            i_all_red = grid_all.get_red_index(i)
            i_marg_red = grid_marg.get_red_index(i_marg)
            
            x_marg[i_marg_red] += x[i_all_red]
            

        for i in grid_marg.indices():
            
            i_red = grid_marg.get_red_index(i)
            coord = grid_marg.get_coords(i)
            
            f = x_marg[i_red]
            
            for j in i:
                print(" {:4d}".format(j), end=sep, file=file)
            
            for y in coord:
                print(" {:25.15f}".format(y), end=sep, file=file)
            
            print(" {:25.15f}".format(f), file=file)

