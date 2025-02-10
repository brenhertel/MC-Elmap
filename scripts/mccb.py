import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from scipy.optimize import minimize

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './Guassian-Mixture-Models')
from GMM_GMR import GMM_GMR

from utils import *

class MCCB(object):

    def __init__(self, trajs, weighting='auto', spatial=0.33, tangent=0.33, shape=0.33):
        self.trajs = trajs
        self.n_trajs = len(self.trajs)
        self.n_pts, self.n_dims = np.shape(trajs[0])
        self.last_x = None
        
        self.weighting_scheme = weighting
        self.spatial_alpha = spatial
        self.tangent_alpha = tangent
        self.laplacian_alpha = shape
        self.spatial_beta = 1
        self.tangent_beta = 1
        self.laplacian_beta = 1
        self.calc_params()
        
        self.trajs_with_time = []
        self.t = np.linspace(0, 1, self.n_pts).reshape((self.n_pts, 1))
        for i in range(self.n_trajs):
            self.trajs_with_time.append(np.hstack((self.t, self.trajs[i])))
            
        self.calc_tangent()
        self.calc_laplacian()
        self.encode_GMMs()
        
    def calc_params(self):
        self.spatial_const = self.spatial_alpha / self.spatial_beta
        self.tangent_const = self.tangent_alpha / self.tangent_beta
        self.laplacian_const = self.laplacian_alpha / self.laplacian_beta
        
    def calc_tangent(self):
        e1 = np.diag(-1*np.ones(self.n_pts-1), -1)
        e2 = np.diag(np.ones(self.n_pts))
        self.E = np.zeros((self.n_pts, self.n_pts))
        self.E += e1
        self.E += e2
        
        self.E_cvx = np.zeros((self.n_pts*self.n_dims, self.n_pts*self.n_dims))
        for i in range(self.n_dims):
            self.E_cvx[self.n_pts*i : self.n_pts*(i + 1), self.n_pts*i : self.n_pts*(i + 1)] = self.E
        
        self.t_trajs = []
        self.t_trajs_with_time = []
        for i in range(self.n_trajs):
            self.t_trajs.append(self.E @ self.trajs[i])
            self.t_trajs_with_time.append(np.hstack((self.t, self.t_trajs[i])))
            
        
    def calc_laplacian(self):
        r1 = 0.5*np.diag(np.ones(self.n_pts-1))
        r2 = -1*np.diag(np.ones(self.n_pts))
        self.R = np.zeros((self.n_pts, self.n_pts))
        self.R[1:,0:self.n_pts-1] += r1
        self.R += r2
        self.R[:-1,1:self.n_pts] += r1
        
        self.R_cvx = np.zeros((self.n_pts*self.n_dims, self.n_pts*self.n_dims))
        for i in range(self.n_dims):
            self.R_cvx[self.n_pts*i : self.n_pts*(i + 1), self.n_pts*i : self.n_pts*(i + 1)] = self.R
        
        self.l_trajs = []
        self.l_trajs_with_time = []
        for i in range(self.n_trajs):
            self.l_trajs.append(self.R @ self.trajs[i])
            self.l_trajs_with_time.append(np.hstack((self.t, self.l_trajs[i])))
        
    def encode_GMMs(self, num_states=5):
        self.n_states = num_states
        #spatial gmm
        self.s_gmm = GMM_GMR(num_states)
        self.s_gmm.fit(np.transpose(np.vstack(self.trajs_with_time)))
        self.s_gmm.predict(np.transpose(self.t))
        self.mu_s = np.transpose(self.s_gmm.getPredictedData())
        self.cov_s = self.s_gmm.getPredictedSigma()
        self.inv_cov_s = np.zeros((np.shape(self.cov_s)))
        for i in range(self.n_pts):
            self.inv_cov_s[:, :, i] = np.linalg.inv(self.cov_s[:, :, i])
        #tangent gmm
        self.t_gmm = GMM_GMR(num_states)
        self.t_gmm.fit(np.transpose(np.vstack(self.t_trajs_with_time)))
        self.t_gmm.predict(np.transpose(self.t))
        self.mu_t = np.transpose(self.t_gmm.getPredictedData())
        self.cov_t = self.t_gmm.getPredictedSigma()
        self.inv_cov_t = np.zeros((np.shape(self.cov_t)))
        for i in range(self.n_pts):
            self.inv_cov_t[:, :, i] = np.linalg.inv(self.cov_t[:, :, i])
        #laplacian gmm
        self.l_gmm = GMM_GMR(num_states)
        self.l_gmm.fit(np.transpose(np.vstack(self.l_trajs_with_time)))
        self.l_gmm.predict(np.transpose(self.t))
        self.mu_l = np.transpose(self.l_gmm.getPredictedData())
        self.cov_l = self.l_gmm.getPredictedSigma()
        self.inv_cov_l = np.zeros((np.shape(self.cov_l)))
        for i in range(self.n_pts):
            self.inv_cov_l[:, :, i] = np.linalg.inv(self.cov_l[:, :, i])
        return
        
    def calc_scaling_factors(self):
        error_s = 0
        error_t = 0
        error_l = 0
        for i in range(self.n_trajs):
            for n in range(self.n_pts):
                diff_s = self.trajs[i][n, :] - self.mu_s[n, 1:]
                error_s = error_s + abs((diff_s @ self.inv_cov_s[:, :, n]) @ np.transpose(diff_s))
                diff_t = self.t_trajs[i][n, :] - self.mu_t[n, 1:]
                error_t = error_t + abs((diff_t @ self.inv_cov_t[:, :, n]) @ np.transpose(diff_t))
                diff_l = self.l_trajs[i][n, :] - self.mu_l[n, 1:]
                error_l = error_l + abs((diff_t @ self.inv_cov_l[:, :, n]) @ np.transpose(diff_l))
        if error_s <= 0 or error_t <= 0 or error_l <= 0:
            print('WARNING WARNING ERROR LESS THAN 0 WARNING WARNING')
        sum_error = error_s + error_t + error_l
        self.spatial_beta = error_s / sum_error
        self.tangent_beta = error_t / sum_error
        self.laplacian_beta = error_l / sum_error
        print("Betas found: " + str([self.spatial_beta, self.tangent_beta, self.laplacian_beta]))
        
    def cost_balancing_objective(self, X):
        x = cp.Variable(np.shape(self.mu_s_stacked))
        if self.last_x is not None:
            x.value = self.last_x
            
        consts = []
        for i in range(len(self.cst_inds)):
            for j in range(self.n_dims):
                consts.append( cp.abs(x[(self.cst_inds[i] * self.n_dims) + j][0] - self.cst_pts[i][j]) <= 0 )
                
        #objective = cp.Minimize((X[0] / self.spatial_beta) * ((self.mu_s_stacked - x).T @ self.cov_s_stacked @ (self.mu_s_stacked - x)) + (X[1] / self.tangent_beta) * (self.mu_t_stacked - self.E_cvx @ x).T @ self.cov_t_stacked @ (self.mu_t_stacked - self.E_cvx @ x) + (X[2] / self.laplacian_beta) * (self.mu_l_stacked - self.R_cvx @ x).T @ self.cov_l_stacked @ (self.mu_l_stacked - self.R_cvx @ x))
        objective = cp.Minimize( (X[0] / self.spatial_beta) * cp.sum_squares(self.cov_s_cholesky @ (x - self.mu_s_stacked)) + (X[1] / self.tangent_beta) * cp.sum_squares(self.cov_t_cholesky @ (self.E_cvx @ x - self.mu_t_stacked)) + (X[2] / self.laplacian_beta) * cp.sum_squares(self.cov_l_cholesky @ (self.R_cvx @ x - self.mu_l_stacked)))
        problem = cp.Problem(objective, consts)
        if self.last_x is not None:
            problem.solve(solver=cp.OSQP, verbose=False, warm_start=True, max_iter=1000000, eps_abs=1e-3, eps_rel=1e-3)
        else:
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=1000000, eps_abs=1e-3, eps_rel=1e-3)
        repro = np.reshape(x.value, (self.n_pts, self.n_dims))
        self.last_x = x.value
        J = 0
        for i in range(self.n_trajs):
            J = J + np.linalg.norm(repro - self.trajs[i])
        return J
        
    def cost_balancing(self):
        res = minimize(self.cost_balancing_objective, np.array([self.spatial_alpha, self.tangent_alpha, self.laplacian_alpha]), method='SLSQP', bounds=((0, 1), (0, 1), (0, 1)), constraints=({'type': 'eq', 'fun' : lambda x: sum(x) - 1}), options={'disp' : True})
        print("Alphas found: " + str(res.x))
        self.spatial_alpha = res.x[0]
        self.tangent_alpha = res.x[1]
        self.laplacian_alpha = res.x[2]
        return
        
    def get_repro(self):
        self.mu_s_stacked = np.reshape(self.mu_s[:, 1:], (self.n_pts*self.n_dims, 1))
        self.mu_t_stacked = np.reshape(self.mu_t[:, 1:], (self.n_pts*self.n_dims, 1))
        self.mu_l_stacked = np.reshape(self.mu_l[:, 1:], (self.n_pts*self.n_dims, 1))
        
        self.cov_s_stacked = np.zeros((self.n_dims * self.n_pts, self.n_dims * self.n_pts))
        for i in range(self.n_pts):
            self.cov_s_stacked[self.n_dims*i : self.n_dims*(i+1), self.n_dims*i : self.n_dims*(i+1)] = self.inv_cov_s[:, :, i]
        self.cov_s_cholesky = np.linalg.cholesky(self.cov_s_stacked)
        self.cov_t_stacked = np.zeros((self.n_dims * self.n_pts, self.n_dims * self.n_pts))
        for i in range(self.n_pts):
            self.cov_t_stacked[self.n_dims*i : self.n_dims*(i+1), self.n_dims*i : self.n_dims*(i+1)] = self.inv_cov_t[:, :, i]
        self.cov_t_cholesky = np.linalg.cholesky(self.cov_t_stacked)
        self.cov_l_stacked = np.zeros((self.n_dims * self.n_pts, self.n_dims * self.n_pts))
        for i in range(self.n_pts):
            self.cov_l_stacked[self.n_dims*i : self.n_dims*(i+1), self.n_dims*i : self.n_dims*(i+1)] = self.inv_cov_l[:, :, i]
        self.cov_l_cholesky = np.linalg.cholesky(self.cov_l_stacked)
        
        if self.weighting_scheme == 'auto':
            self.calc_scaling_factors()
            self.cost_balancing()
            self.calc_params()
        
        self.x = cp.Variable(np.shape(self.mu_s_stacked))
        if self.last_x is not None:
            self.x.value = self.last_x
        #self.objective = cp.Minimize(self.spatial_const * (self.mu_s_stacked - self.x).T @ self.cov_s_stacked @ (self.mu_s_stacked - self.x) + self.tangent_const * (self.mu_t_stacked - self.E_cvx @ self.x).T @ self.cov_t_stacked @ (self.mu_t_stacked - self.E_cvx @ self.x) + self.laplacian_const * (self.mu_l_stacked - self.R_cvx @ self.x).T @ self.cov_l_stacked @ (self.mu_l_stacked - self.R_cvx @ self.x))
        self.objective = cp.Minimize( self.spatial_const * cp.sum_squares(self.cov_s_cholesky @ (self.x - self.mu_s_stacked)) + self.tangent_const * cp.sum_squares(self.cov_t_cholesky @ (self.E_cvx @ self.x - self.mu_t_stacked)) + self.laplacian_const * cp.sum_squares(self.cov_l_cholesky @ (self.R_cvx @ self.x - self.mu_l_stacked)))
        self.calc_consts()
        self.problem = cp.Problem(self.objective, self.consts)
        if self.last_x is not None:
            self.problem.solve(solver=cp.OSQP, verbose=True, warm_start=True, max_iter=1000000)
        else:
            self.problem.solve(solver=cp.OSQP, verbose=True, max_iter=1000000)
        
        repro = np.reshape(self.x.value, (self.n_pts, self.n_dims))
        #x_inv, _, _, _ = np.linalg.lstsq(self.R_cvx, self.mu_l_stacked, rcond=-1)
        #repro = np.reshape(x_inv, (self.n_pts, self.n_dims))
        
        return repro
        
    def set_constraints(self, inds=[], csts=[]):
        self.cst_inds = inds
        self.cst_pts = csts
        
    def calc_consts(self):
        self.consts = []
        for i in range(len(self.cst_inds)):
            for j in range(self.n_dims):
                self.consts.append( cp.abs(self.x[(self.cst_inds[i] * self.n_dims) + j][0] - self.cst_pts[i][j]) <= 0 )
        
        
def main_lasa():
    trajs = []
    for i in range(1, 8):
        trajs.append(get_lasa_trajN('heee', n=i))
        
        
    cst_inds = [0, len(trajs[0])-1]
    cst_pts = [trajs[0][0, :], trajs[0][-1, :]]
    mc = MCCB(trajs, weighting='custom', spatial=0.33, shape=0.33, tangent=0.33)
    #x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.get_repro()
            
    plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k')
    plt.plot(repro[:, 0], repro[:, 1], 'r')
    for i in range(len(cst_inds)):
        plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
        plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    plt.figure()
    plt.title('x')
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 0], 'k')
    plt.plot(repro[:, 0], 'r')
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], cst_pts[i][0], 'k.', ms=15)
        plt.plot(cst_inds[i], repro[cst_inds[i], 0], 'r.', ms=15)
    plt.figure()
    plt.title('y')
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 1], 'k')
    plt.plot(repro[:, 1], 'r')
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], cst_pts[i][1], 'k.', ms=15)
        plt.plot(cst_inds[i], repro[cst_inds[i], 1], 'r.', ms=15)
        
    plt.show()

if __name__ == '__main__':
    main_lasa()