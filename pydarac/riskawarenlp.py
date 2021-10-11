# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:37:59 2021

@author: Jukka Isohatala
"""

import numpy as np
import cyipopt as ip
import scipy.sparse as sparse

from riskfunction import *


class RiskAwareControlNLP(ip.Problem):
    """Base class for risk-aware stochastic control problems, based
    on the dynamic analytic method of Isohatala and Haskell. The control
    problem is solved by minimizing the given risk function over
    time-dependent distributions of the controlled process. This
    measure-valued process is given by the forward equation, which
    here is solved in some discretized, approximate form.

    """


    def __init__(self,
                 n_disc_tot_x, n_disc_tot_xa, 
                 n_disc_y, n_disc_q,
                 n_disc_t,
                 risk_func):
        """Initialize the problem object with n_disc_tot_x numbers of
        X-space total sample points, n_disc_tot_xa total X*A-space
        sample points (A-space sample points not given separately
        as a parameter, since we may want to constrain the possible
        A values by the corresponding X-coordinate; hence X*A-space
        points may be different from a simple product), n_disc_y
        is the number of discretization points on the running cost space,
        and n_disc_q is the number of points on the total cost space
        (finall running costs plus a possible terminal cost).
        The argument risk_func is the used risk function, a RiskFunction
        instance.

        """

        self.n_disc_tot_x = n_disc_tot_x
        self.n_disc_tot_xa = n_disc_tot_xa
        self.n_disc_y = n_disc_y
        self.n_disc_q = n_disc_q
        self.n_disc_t = n_disc_t
        self.n_rf_pars = risk_func.get_n_infpars()

        self.n_xyt_constraints = n_disc_tot_x * n_disc_y * n_disc_t
        self.n_constraints = self.n_xyt_constraints + self.n_disc_q
        self.n_xyat_vars = n_disc_tot_xa * n_disc_y * n_disc_t
        self.n_vars = self.n_xyat_vars + n_disc_q + self.n_rf_pars

        self.risk_func = risk_func

        # Start constructing the linear constraints.
        # First, setup for the forward equation part:
        A = sparse.dok_matrix((self.n_xyt_constraints,
                               self.n_xyat_vars))
        b = np.zeros(self.n_xyt_constraints)
        ic = self.get_ic()

        if len(ic) != n_disc_tot_x * n_disc_y:
            raise ValueError("The length of the initial condition vector "
                             + "should be n_xyat_vars")

        self.setup_A_b(A, b, ic)
        
        # Second, augment the forward equation with the projection
        # operator, mapping a solution vector onto a cost distribution
        P = sparse.dok_matrix((self.n_disc_q, self.n_xyat_vars))
        self.setup_q_projection(P)

        # Add the projection operator into A, and resize to accommodate
        # risk function parameters at the end of the solution vector.
        A = sparse.vstack([A, P])
        A.resize((self.n_constraints, self.n_vars))
        A = A.todok()
        b.resize(self.n_constraints)

        # The projection of the solution vector should equal
        # the cost distribution contained in the solution vector itself:
        for i in range(n_disc_q):
            A[self.n_xyt_constraints + i, self.n_xyat_vars + i] = -1.0

        # Done.
        A = A.tocoo()
        self.A_nz_rows = A.row
        self.A_nz_cols = A.col
        self.A_vals = A.data

        self.A = A.tocsr()

        # Precompute Hessian structure
        self.n_hess_nz = self.n_disc_q*(self.n_disc_q + 1) // 2
        self.hess_rows = np.zeros(self.n_hess_nz, dtype=np.dtype(int))
        self.hess_cols = np.zeros(self.n_hess_nz, dtype=np.dtype(int))

        offset = self.n_xyat_vars

        hess_i = 0
        for hess_row in range(self.n_disc_q):
            for hess_col in range(hess_row + 1):
                self.hess_rows[hess_i] = offset + hess_row
                self.hess_cols[hess_i] = offset + hess_col
                hess_i += 1

        
        # TODO: This is in general incorrect when n_rf_infpars > 0
        lb = np.zeros(self.n_vars)
        super(RiskAwareControlNLP, self).__init__(n = self.n_vars,
                                                  m = self.n_constraints,
                                                  cl = b,
                                                  cu = b,
                                                  lb = lb)   

        # self.add_option('print_level', 12)
        # self.add_option('derivative_test', 'second-order')
        # self.add_option('derivative_test_print_all', 'yes')
        # self.add_option('jac_d_constant', 'yes')
        # self.add_option('hessian_constant', 'yes')
        self.add_option('jac_c_constant', 'yes')
        self.add_option('mu_strategy', 'adaptive')
        self.add_option('max_iter', 800)
        self.add_option('tol', 1e-9)


    def get_xyat_part(self, m):
        """Return a view to the X*Y*A*T part of a solution vector."""

        return m[:self.n_xyat_vars]


    def get_cost_distr(self, m):
        """Return a view to the cost distribution part of a solution vector."""

        return m[self.n_xyat_vars:self.n_xyat_vars + self.n_disc_q]


    def get_rf_infpars(self, m):
        """Return a view to the risk function parameter part of a
        solution vector."""

        return m[-self.n_rf_infpars:]


    def get_ic(self):
        """Return the initial distribution for the control problem."""
        
        raise NotImplementedError("get_ic must be implemented")

        
    def get_solver_iv(self):
        """Return an inital guess for the full solution vector."""
        
        return np.ones(self.n_vars) / (self.n_disc_tot_xa * self.n_disc_y)
        

    def setup_A_b(self, A, b, ic):
        """Setup the forward operator matrix into the matrix A and
        the associated constraint r.h.s. vector as b.
        Only the spatial part is set here, that is, the portion of
        the constraint that includes the (discretized) forward
        equation along with the initial value.

        The matrix A is passed in in some coordinate sparse representation,
        and has the shape (M, N) where M = n_disc_tot_x * n_disc_y * n_disc_t
        and N = n_disc_tot_xa * n_disc_y * n_disc_t. The vector b is of
        length N. The initial condition is passed in as the vector ic.

        Setting up other constraints required in the solution of
        the problem are done elsewhere.

        """

        raise NotImplementedError("setup_A_b must be implemented")


    def setup_projection(self, P):
        """Fill in the projection matrix P that maps the solution
        vector onto the final time y-distribution. The matrix P
        is an (n_disc_q, n_disc_tot_xa * n_disc_y * n_disc_t)
        COO sparse matrix. This method should handle both
        the running cost and terminal cost.

        """

        raise NotImplementedError("setup_q_projection must be implented")


    def objective(self, m):
        """Evaluate the objective function."""

        if self.n_rf_pars > 0:
            u = self.get_rf_infpars(m)
        else:
            u = None
            
        cost_distr = self.get_cost_distr(m)

        return self.risk_func.eval(cost_distr, u=u)


    def constraints(self, m):
        """Evaluate the constraints, here simply contained in the matrix A."""

        return self.A.dot(m)


    def jacobianstructure(self):

        return self.A_nz_rows, self.A_nz_cols


    def jacobian(self, m):
        """Jacobian of the constraints, here of course
        just the A operator.

        """

        return self.A_vals


    def hessianstructure(self):

        return self.hess_rows, self.hess_cols


    def gradient(self, m):
        """Evaluate the gradient of the objective."""

        if self.n_rf_pars > 0:
            u = self.get_rf_infpars(m)
        else:
            u = None
            
        cost_distr = self.get_cost_distr(m)

        grad = np.zeros(self.n_vars)
        grad[self.n_xyat_vars:(self.n_xyat_vars 
                               + self.n_disc_q + self.n_rf_pars)] \
            = self.risk_func.eval_grad(cost_distr, u=u)

        return grad


    def hessian(self, m, lagrange, obj_factor):
        """Evaluate the Hessian of the objective."""

        if self.n_rf_pars > 0:
            u = self.get_rf_infpars(m)
        else:
            u = None
            
        cost_distr = self.get_cost_distr(m)

        rf_hess = self.risk_func.eval_hess(cost_distr, u=u)

        hess = np.zeros(self.n_hess_nz)

        hess_i = 0
        for hess_row in range(self.n_disc_q + self.n_rf_pars):
            for hess_col in range(hess_row + 1):
                hess[hess_i] = obj_factor * rf_hess[hess_row, hess_col]
                hess_i += 1

        return hess


    def solve(self, iv=None):
        """Solve the problem."""

        if iv is None:
            iv = self.get_solver_iv()

        x, info = super(RiskAwareControlNLP, self).solve(iv)

        return x, info


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
