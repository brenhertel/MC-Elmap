import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, NonlinearConstraint

from utils import *
from downsampling import *

def zip_demos(trajs):
    num_demos = len(trajs)
    traj_n, n_dim = np.shape(trajs[0])
    out = np.zeros((traj_n*num_demos, n_dim))
    idx = 0
    for i in range(traj_n):
        for j in range(num_demos):
            out[idx, :] = trajs[j][i, :]
            idx = idx + 1
    return out

def cluster(x, y):
    n_pts, n_dims = np.shape(x)
    n_nodes, _ = np.shape(y)
    clusters = [[] for j in range(n_nodes)]
    # cluster to closest node
    for i in range(n_pts):
        dist = float('inf')
        idx = -1
        for j in range(n_nodes):
            new_dist = np.linalg.norm(x[i] - y[j])
            if new_dist < dist:
                dist = new_dist
                idx = j
        if idx != -1:
            #print(i, j, dist)
            clusters[idx].append(i)
        else:
            print('No Cluster Found!!!')
    return clusters
        
    
        
class MC_ElMap(object):

    def __init__(self, trajs, n=100, weighting='auto', spatial=1.0, shape=1.0, tangent=1.0, stretch=0.01, bend=0.001, init_nodes=None, ds=False):
        self.trajs = trajs
        self.tgt_traj = np.vstack(trajs)
        self.n_pts, self.n_dims = np.shape(self.tgt_traj)
        self.n_pts2 = 2*self.n_pts
        self.n_nodes = n
        self.n_nodes2 = n*2
        self.n_size = self.n_nodes * self.n_dims
        self.clusters = []
        self.weighting_scheme = weighting
        self.traj_len = len(self.trajs[0])
        self.num_trajs = len(self.trajs)
        if init_nodes is None:
            self.nodes = DouglasPeuckerPoints(self.trajs[0], self.n_nodes)
        else:
            self.nodes = init_nodes
        self.ds = ds
        if self.ds:
            self.traj_ds = DouglasPeuckerPoints(self.trajs[0], self.n_nodes)
            
        if self.n_dims == 1:
            self.traj_x = self.tgt_traj
            self.traj_stacked = self.traj_x
            self.nodes_stacked = self.nodes
            if self.ds:
                self.traj_ds_stacked = self.traj_ds
        elif self.n_dims == 2:
            self.traj_x = np.reshape(self.tgt_traj[:, 0], (self.n_pts, 1))
            self.traj_y = np.reshape(self.tgt_traj[:, 1], (self.n_pts, 1))
            self.traj_stacked = np.vstack((self.traj_x, self.traj_y))
            nodes_x = np.reshape(self.nodes[:, 0], (self.n_nodes, 1))
            nodes_y = np.reshape(self.nodes[:, 1], (self.n_nodes, 1))
            self.nodes_stacked = np.vstack((nodes_x, nodes_y))
            if self.ds:
                traj_ds_x = np.reshape(self.traj_ds[:, 0], (self.n_nodes, 1))
                traj_ds_y = np.reshape(self.traj_ds[:, 1], (self.n_nodes, 1))
                self.traj_ds_stacked = np.vstack((traj_ds_x, traj_ds_y))
        elif self.n_dims == 3:
            self.traj_x = np.reshape(self.tgt_traj[:, 0], (self.n_pts, 1))
            self.traj_y = np.reshape(self.tgt_traj[:, 1], (self.n_pts, 1))
            self.traj_z = np.reshape(self.tgt_traj[:, 2], (self.n_pts, 1))
            self.traj_stacked = np.vstack((self.traj_x, self.traj_y, self.traj_z))
            nodes_x = np.reshape(self.nodes[:, 0], (self.n_nodes, 1))
            nodes_y = np.reshape(self.nodes[:, 1], (self.n_nodes, 1))
            nodes_z = np.reshape(self.nodes[:, 2], (self.n_nodes, 1))
            self.nodes_stacked = np.vstack((nodes_x, nodes_y, nodes_z))
            if self.ds:
                traj_ds_x = np.reshape(self.traj_ds[:, 0], (self.n_nodes, 1))
                traj_ds_y = np.reshape(self.traj_ds[:, 1], (self.n_nodes, 1))
                traj_ds_z = np.reshape(self.traj_ds[:, 2], (self.n_nodes, 1))
                self.traj_ds_stacked = np.vstack((traj_ds_x, traj_ds_y, traj_ds_z))
        else:
            print("Too many dimensions! n_dims must be <= 3")
            exit()
            
            
        e1 = np.diag(-1*np.ones(self.n_size-1), -1)
        e2 = np.diag(np.ones(self.n_size))
        self.E = np.zeros((self.n_size, self.n_size))
        self.E += e1
        self.E += e2
        if self.n_dims >= 2:
            self.E[self.n_nodes, self.n_nodes-1] = 0
            self.E[self.n_nodes, self.n_nodes] = 0
        if self.n_dims == 3:
            self.E[self.n_nodes2, self.n_nodes2-1] = 0
            self.E[self.n_nodes2, self.n_nodes2] = 0
            
        r1 = np.diag(np.ones(self.n_size-1))
        r2 = -2*np.diag(np.ones(self.n_size))
        self.R = np.zeros((self.n_size, self.n_size))
        self.R[1:,0:self.n_size-1] += r1
        self.R += r2
        self.R[:-1,1:self.n_size] += r1
        if self.n_dims >= 2:
            self.R[self.n_nodes, self.n_nodes+1] = 0
            self.R[self.n_nodes, self.n_nodes-1] = 0
            self.R[self.n_nodes, self.n_nodes] = 0
            self.R[self.n_nodes-1, self.n_nodes-1] = 0
            self.R[self.n_nodes-1, self.n_nodes] = 0
            self.R[self.n_nodes-1, self.n_nodes-2] = 0
        if self.n_dims == 3:
            self.R[self.n_nodes2, self.n_nodes2+1] = 0
            self.R[self.n_nodes2, self.n_nodes2-1] = 0
            self.R[self.n_nodes2, self.n_nodes2] = 0
            self.R[self.n_nodes2-1, self.n_nodes2-1] = 0
            self.R[self.n_nodes2-1, self.n_nodes2] = 0
            self.R[self.n_nodes2-1, self.n_nodes2-2] = 0  
            
        self.spatial_alpha = spatial
        self.spatial_beta = 1.0
        
        self.tangent_alpha = tangent
        self.tangent_beta = 1.0
        
        self.shape_alpha = stretch
        self.shape_beta = 1.0
        
        self.calc_params()
        
        self.stretch_const = stretch
        self.bend_const = bend
        if not self.ds:
            self.convert_Tangent()
            self.convert_Laplacian()
        self.set_constraints()
        
    def calc_params(self):
        self.spatial_const = self.spatial_alpha / self.spatial_beta
        self.tangent_const = self.tangent_alpha / self.tangent_beta
        self.shape_const = self.shape_alpha / self.shape_beta
        
        
    def convert_Tangent(self):
        
        e1 = np.diag(-1*np.ones(self.n_pts-1), -1)
        e2 = np.diag(np.ones(self.n_pts))
        E = np.zeros((self.n_pts, self.n_pts))
        E += e1
        E += e2
        
        for i in range(self.num_trajs):
            E[i*self.traj_len, i*self.traj_len] = 0
            E[i*self.traj_len, i*self.traj_len - 1 ] = 0
            
        self.traj_tan = E @ self.tgt_traj
        
        if self.n_dims == 1:
            self.traj_x_tan = self.traj_tan
            self.traj_stacked_tan = self.traj_x_tan
        elif self.n_dims == 2:
            self.traj_x_tan = np.reshape(self.traj_tan[:, 0], (self.n_pts, 1))
            self.traj_y_tan = np.reshape(self.traj_tan[:, 1], (self.n_pts, 1))
            self.traj_stacked_tan = np.vstack((self.traj_x_tan, self.traj_y_tan))
        elif self.n_dims == 3:
            self.traj_x_tan = np.reshape(self.traj_tan[:, 0], (self.n_pts, 1))
            self.traj_y_tan = np.reshape(self.traj_tan[:, 1], (self.n_pts, 1))
            self.traj_z_tan = np.reshape(self.traj_tan[:, 2], (self.n_pts, 1))
            self.traj_stacked_tan = np.vstack((self.traj_x_tan, self.traj_y_tan, self.traj_z_tan))
        else:
            print("Too many dimensions! n_dims must be <= 3")
            exit()
        return
        
    def convert_Laplacian(self):
            
        r1 = np.diag(np.ones(self.n_pts-1))
        r2 = -2*np.diag(np.ones(self.n_pts))
        R = np.zeros((self.n_pts, self.n_pts))
        R[1:,0:self.n_pts-1] += r1
        R += r2
        R[:-1,1:self.n_pts] += r1
        
        for i in range(self.num_trajs):
            R[i*self.traj_len, i*self.traj_len+1] = 0
            R[i*self.traj_len, i*self.traj_len-1] = 0
            R[i*self.traj_len, i*self.traj_len] = 0
            R[i*self.traj_len-1, i*self.traj_len-1] = 0
            R[i*self.traj_len-1, i*self.traj_len] = 0
            R[i*self.traj_len-1, i*self.traj_len-2] = 0
            
        self.traj_lap = R @ self.tgt_traj
        
        if self.n_dims == 1:
            self.traj_x_lap = self.traj_lap
            self.traj_stacked_lap = self.traj_x_lap
        elif self.n_dims == 2:
            self.traj_x_lap = np.reshape(self.traj_lap[:, 0], (self.n_pts, 1))
            self.traj_y_lap = np.reshape(self.traj_lap[:, 1], (self.n_pts, 1))
            self.traj_stacked_lap = np.vstack((self.traj_x_lap, self.traj_y_lap))
        elif self.n_dims == 3:
            self.traj_x_lap = np.reshape(self.traj_lap[:, 0], (self.n_pts, 1))
            self.traj_y_lap = np.reshape(self.traj_lap[:, 1], (self.n_pts, 1))
            self.traj_z_lap = np.reshape(self.traj_lap[:, 2], (self.n_pts, 1))
            self.traj_stacked_lap = np.vstack((self.traj_x_lap, self.traj_y_lap, self.traj_z_lap))
        else:
            print("Too many dimensions! n_dims must be <= 3")
            exit()
        return
       
    def calc_scaling_factors(self, A, B):
        if self.ds:
            spatial_error = np.linalg.norm(A @ self.nodes_stacked - B @ self.traj_stacked)
            tangent_error = 0
            laplacian_error = 0
            for i in range(self.num_trajs):
                traj_ds = DouglasPeuckerPoints(self.trajs[i], self.n_nodes)
                
                if self.n_dims == 1:
                    traj_ds_stacked = traj_ds
                elif self.n_dims == 2:
                    traj_ds_x = np.reshape(traj_ds[:, 0], (self.n_nodes, 1))
                    traj_ds_y = np.reshape(traj_ds[:, 1], (self.n_nodes, 1))
                    traj_ds_stacked = np.vstack((traj_ds_x, traj_ds_y))
                elif self.n_dims == 3:
                    traj_ds_x = np.reshape(traj_ds[:, 0], (self.n_nodes, 1))
                    traj_ds_y = np.reshape(traj_ds[:, 1], (self.n_nodes, 1))
                    traj_ds_z = np.reshape(traj_ds[:, 2], (self.n_nodes, 1))
                    traj_ds_stacked = np.vstack((traj_ds_x, traj_ds_y, traj_ds_z))
                
                tangent_error = tangent_error + np.linalg.norm(self.E @ self.nodes_stacked - self.E @ traj_ds_stacked)
                laplacian_error = laplacian_error + np.linalg.norm(self.R @ self.nodes_stacked - self.R @ traj_ds_stacked)
        else:
            spatial_error = np.linalg.norm(A @ self.nodes_stacked - B @ self.traj_stacked)
            tangent_error = np.linalg.norm(A @ (self.E @ self.nodes_stacked) - B @ self.traj_stacked_tan)
            laplacian_error = np.linalg.norm(A @ (self.R @ self.nodes_stacked) - B @ self.traj_stacked_lap)
        sum = spatial_error + tangent_error + laplacian_error
        return [spatial_error / sum, tangent_error / sum, laplacian_error / sum]
      
    def smoothing_autotuning(self, A, B):
        if self.ds:
            approx_cost = self.spatial_const * np.linalg.norm(A @ self.nodes_stacked - B @ self.traj_stacked) + self.tangent_const * np.linalg.norm(self.E @ self.nodes_stacked - self.E @ self.traj_ds_stacked) + self.shape_const * np.linalg.norm(self.R @ self.nodes_stacked - self.R @ self.traj_ds_stacked)
        else:
            approx_cost = self.spatial_const * np.linalg.norm(A @ self.nodes_stacked - B @ self.traj_stacked) + self.tangent_const * np.linalg.norm(A @ (self.E @ self.nodes_stacked) - B @ self.traj_stacked_tan) + self.shape_const * np.linalg.norm(A @ (self.R @ self.nodes_stacked) - B @ self.traj_stacked_lap)
        stretch_cost = np.linalg.norm(self.E @ self.nodes_stacked)
        bend_cost = np.linalg.norm(self.R @ self.nodes_stacked)
        
        #run1
        #stretch_mult = 100.0
        #bend_mult = 1000.0
        
        #run 2, 3
        #stretch_mult = 100.0
        #bend_mult = 100.0
        
        #run 4
        stretch_mult = 100.0
        bend_mult = 200.0
        
        self.stretch_const = stretch_mult * approx_cost / stretch_cost
        self.bend_const = bend_mult * approx_cost / bend_cost
        if approx_cost <= 0:
            self.stretch_const = 1
            self.bend_const = 1
        print("Smoothing Parameters: " + str([self.stretch_const, self.bend_const]))
        return
      
    def calc_parameters(self, x, A, B):
        #(scaling_factors, A, B, AT, BT, AL, BL) = args
        x_repro = cp.Variable((self.n_size, 1))
        if self.ds:
            objective = cp.Minimize((x[0] / self.spatial_beta) * cp.sum_squares(A @ x_repro - B @ self.traj_stacked) 
                                     + (x[1] / self.tangent_beta) * cp.sum_squares(self.E @ x_repro - self.E @ self.traj_ds_stacked) 
                                     + (x[2] / self.shape_beta) * cp.sum_squares(self.R @ x_repro - self.R @ self.traj_ds_stacked) 
                                     + self.stretch_const * cp.sum_squares(self.E @ x_repro)
                                     + self.bend_const    * cp.sum_squares(self.R @ x_repro))
        else:
            objective = cp.Minimize((x[0] / self.spatial_beta) * cp.sum_squares(A @ x_repro - B @ self.traj_stacked) 
                                     + (x[1] / self.tangent_beta) * cp.sum_squares(A @ (self.E @ x_repro) - B @ self.traj_stacked_tan) 
                                     + (x[2] / self.shape_beta) * cp.sum_squares(A @ (self.R @ x_repro) - B @ self.traj_stacked_lap) 
                                     + self.stretch_const * cp.sum_squares(self.E @ x_repro)
                                     + self.bend_const    * cp.sum_squares(self.R @ x_repro))
        problem = cp.Problem(objective)
        problem.solve()
        if x_repro.value is not None:
            repro = x_repro.value
            #print(np.shape(AL), np.shape(self.R2), np.shape(repro))
            #print(np.shape((self.R2 @ repro)))
            #print(np.shape(AL @ (self.R2 @ repro)))
            #objective_val = (x[0] / self.spatial_beta) * np.linalg.norm(A @ repro - B @ self.traj_stacked) + (x[1] / self.tangent_beta) * np.linalg.norm(AT @ (self.E2 @ repro) - BT @ self.traj_stacked_tan) + (x[2] / self.shape_beta) * np.linalg.norm(AL @ (self.R2 @ repro) - BL @ self.traj_stacked_lap)
            if self.ds:
                objective_val = (1 / self.spatial_beta) * np.linalg.norm(A @ repro - B @ self.traj_stacked) + (1 / self.tangent_beta) * np.linalg.norm(self.E @ repro - self.E @ self.traj_ds_stacked) + (1 / self.shape_beta) * np.linalg.norm(self.R @ repro - self.R @ self.traj_ds_stacked)
            else:
                objective_val = (1 / self.spatial_beta) * np.linalg.norm(A @ repro - B @ self.traj_stacked) + (1 / self.tangent_beta) * np.linalg.norm(A @ (self.E @ repro) - B @ self.traj_stacked_tan) + (1 / self.shape_beta) * np.linalg.norm(A @ (self.R @ repro) - B @ self.traj_stacked_lap)
            #objective_val = np.linalg.norm(A @ repro - B @ self.traj_stacked) + np.linalg.norm(AT @ (self.E2 @ repro) - BT @ self.traj_stacked_tan) + np.linalg.norm(AL @ (self.R2 @ repro) - BL @ self.traj_stacked_lap)
            # maybe add in smoothing here?
        else:
            print("REPRO IS NONE")
            objective_val = 1e20
        return objective_val
      
    def setup_problem(self):
        self.clusters = cluster(self.tgt_traj, self.nodes)
        A = np.zeros((self.n_nodes*self.n_dims, self.n_nodes*self.n_dims))
        B = np.zeros((self.n_nodes*self.n_dims, self.n_pts*self.n_dims))
        for i in range(self.n_nodes):
            for pt in self.clusters[i]:
                for j in range(self.n_dims):
                    B[i + self.n_nodes*j, pt + self.n_pts*j] = 1 #change for variable weighting
            for j in range(self.n_dims):
                A[i + self.n_nodes*j, i + self.n_nodes*j] = np.sum(B[i + self.n_nodes*j, :])
                
        '''
        AT = np.zeros(((self.n_nodes-1)*self.n_dims, (self.n_nodes-1)*self.n_dims))
        BT = np.zeros(((self.n_nodes-1)*self.n_dims, (self.n_pts-1)*self.n_dims))
        for i in range(1, self.n_nodes):
            for pt in self.clusters[i]:
                for j in range(self.n_dims):
                    #print("i: " + str(i-1 + self.n_nodes*j))
                    #print("j: " + str(pt + self.n_pts*j))
                    BT[i-1 + (self.n_nodes-1)*j, pt -1 + (self.n_pts-1)*j] = 1 #change for variable weighting
            for j in range(self.n_dims):
                AT[i-1 + (self.n_nodes-1)*j, i-1 + (self.n_nodes-1)*j] = np.sum(BT[i-1 + (self.n_nodes-1)*j, :])
                
        AL = np.zeros(((self.n_nodes-2)*self.n_dims, (self.n_nodes-2)*self.n_dims))
        BL = np.zeros(((self.n_nodes-2)*self.n_dims, (self.n_pts-2)*self.n_dims))
        for i in range(1, self.n_nodes-1):
            for pt in self.clusters[i]:
                for j in range(self.n_dims):
                    BL[i-1 + (self.n_nodes-2)*j, pt -2 + (self.n_pts-2)*j] = 1 #change for variable weighting
            for j in range(self.n_dims):
                AL[i-1 + (self.n_nodes-2)*j, i-1 + (self.n_nodes-2)*j] = np.sum(BL[i-1 + (self.n_nodes-2)*j, :])
        '''
        
        if self.weighting_scheme == 'auto':
            # get scaling factors (betas)
            #print("MATRICES")
            #print(np.shape(A), np.shape(B), np.shape(AT), np.shape(BT), np.shape(AL), np.shape(BL))
            scaling_factors = self.calc_scaling_factors(A, B)
            self.spatial_beta = scaling_factors[0]
            self.tangent_beta = scaling_factors[1]
            self.shape_beta = scaling_factors[2]
            
            # get importance (alphas)
            #alphas = cp.Variable((3, 1))
            #alpha_objective = cp.Minimize( (alphas[0] / scaling_factors[0]) * cp.sum_squares(A @ self.nodes_stacked - B @ self.traj_stacked) 
            #                         + (alphas[1] / scaling_factors[1]) * cp.sum_squares(AT @ (self.E2 @ self.nodes_stacked) - BT @ self.traj_stacked_tan) 
            #                         + (alphas[2] / scaling_factors[2]) * cp.sum_squares(AL @ (self.R2 @ self.nodes_stacked) - BL @ self.traj_stacked_lap) 
            #                         + self.stretch_const * cp.sum_squares(self.E @ self.nodes_stacked)
            #                         + self.bend_const    * cp.sum_squares(self.R @ self.nodes_stacked))
            #alpha_prob = cp.Problem(alpha_objective, [cp.sum(alphas) == 1])
            #alpha_prob.solve(verbose=True)
            #print("Alphas found: " + str(alphas.value))
            #self.spatial_const = alphas[0] / scaling_factors[0]
            #self.tangent_const = alphas[1] / scaling_factors[1]
            #self.shape_const = alphas[2] / scaling_factors[2]
            
            print('#################')
            print('### Inner Opt ###')
            print('#################')
        
            res = minimize(self.calc_parameters, np.array([self.spatial_alpha, self.tangent_alpha, self.shape_alpha]), args=((A, B)), method='SLSQP', bounds=((0, 1), (0, 1), (0, 1)), constraints=({'type': 'eq', 'fun' : lambda x: sum(x) - 1}), options={'disp' : True})
            print("Alphas found: " + str(res.x))
            self.spatial_alpha = res.x[0]
            self.tangent_alpha = res.x[1]
            self.shape_alpha = res.x[2]
            self.calc_params()
            self.smoothing_autotuning(A, B)
        if self.weighting_scheme == 'half-auto':
            self.smoothing_autotuning(A, B)
        
        self.x = cp.Variable((self.n_size, 1))
        self.x.value = self.nodes_stacked
        if self.ds:
            self.objective = cp.Minimize(self.spatial_const   * cp.sum_squares(A @ self.x - B @ self.traj_stacked) 
                                     + self.tangent_const * cp.sum_squares(self.E @ self.x - self.E @ self.traj_ds_stacked) 
                                     + self.shape_const   * cp.sum_squares(self.R @ self.x - self.R @ self.traj_ds_stacked) 
                                     + self.stretch_const * cp.sum_squares(self.E @ self.x)
                                     + self.bend_const    * cp.sum_squares(self.R @ self.x))
        else:
            self.objective = cp.Minimize(self.spatial_const   * cp.sum_squares(A @ self.x - B @ self.traj_stacked) 
                                     + self.tangent_const * cp.sum_squares(A @ (self.E @ self.x) - B @ self.traj_stacked_tan) 
                                     + self.shape_const   * cp.sum_squares(A @ (self.R @ self.x) - B @ self.traj_stacked_lap) 
                                     + self.stretch_const * cp.sum_squares(self.E @ self.x)
                                     + self.bend_const    * cp.sum_squares(self.R @ self.x))
        #self.objective = cp.Minimize(self.spatial_const   * cp.sum_squares(A @ self.x - B @ self.traj_stacked) 
        #                             + self.stretch_const * cp.sum_squares(E @ self.x)
        #                             + self.bend_const    * cp.sum_squares(R @ self.x))
        return self.x
        
    def solve_problem(self, iters=10, disp=True):
        
        for i in range(iters):
            print('#################')
            print('### Outer Opt ###')
            print('#################')
            
            self.calc_consts()
            self.problem = cp.Problem(self.objective, self.consts)
            #print("PROBLEM DATA")
            #print(self.problem.get_problem_data(cp.OSQP))
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=disp, max_iter=1000000)
            
            if disp:
                print("status:", self.problem.status)
                print("optimal value", self.problem.value)
                #print("optimal var", x.value)
                for i in range(len(self.consts)):
                    print("dual value for constraint " + str(i), ": ", self.consts[i].dual_value)
                
            self.solx = self.x
            if self.n_dims == 1:
                self.sol = self.x.value
            elif self.n_dims == 2:
                self.sol = np.hstack((self.x.value[:self.n_nodes], self.x.value[self.n_nodes:]))
            elif self.n_dims == 3:
                self.sol = np.hstack((self.x.value[:self.n_nodes], self.x.value[self.n_nodes:self.n_nodes2], self.x.value[self.n_nodes2:]))
                
            #next round with updated solution
            self.nodes = self.sol
            
            if self.n_dims == 1:
                self.nodes_stacked = self.nodes
            elif self.n_dims == 2:
                nodes_x = np.reshape(self.nodes[:, 0], (self.n_nodes, 1))
                nodes_y = np.reshape(self.nodes[:, 1], (self.n_nodes, 1))
                self.nodes_stacked = np.vstack((nodes_x, nodes_y))
            elif self.n_dims == 3:
                nodes_x = np.reshape(self.nodes[:, 0], (self.n_nodes, 1))
                nodes_y = np.reshape(self.nodes[:, 1], (self.n_nodes, 1))
                nodes_z = np.reshape(self.nodes[:, 2], (self.n_nodes, 1))
                self.nodes_stacked = np.vstack((nodes_x, nodes_y, nodes_z))
            else:
                print("Too many dimensions! n_dims must be <= 3")
                exit()
            
            self.setup_problem()
        return self.sol
        
    def set_constraints(self, inds=[], csts=[]):
        self.cst_inds = inds
        self.cst_pts = csts
        
    def calc_consts(self):
        self.consts = []
        for i in range(len(self.cst_inds)):
            for j in range(self.n_dims):
                self.consts.append( cp.abs(self.x[self.cst_inds[i] + (self.n_nodes * j)][0] - self.cst_pts[i][j]) <= 0 )
        
    def plot_solved_problem(self):
        if self.n_dims == 1:
            fig = plt.figure()
            plt.plot(self.tgt_traj, 'k', lw=5, label="Demo")
            plt.plot(self.sol, 'r', lw=5, label="Repro")
            return fig
        if self.n_dims == 2:
            fig = plt.figure()
            plt.plot(self.traj_x, self.traj_y, 'k', lw=3, label="Demo")
            plt.plot(self.sol[:, 0], self.sol[:, 1], 'r', lw=3, label="Repro")
            return fig
        if self.n_dims == 3:
            print("3D PLOTTING NOT IMPLEMENTED YET")
        return
        
def main_lasa():
    trajs = []
    for i in range(1, 8):
        trajs.append(get_lasa_trajN('Angle', n=i))
        
        
    #traj_stacked = zip_demos(trajs)
    traj_stacked = np.vstack(trajs)
    cst_inds = [0, 99]
    cst_pts = [traj_stacked[0, :], traj_stacked[-1, :]]
    init = DouglasPeuckerPoints(trajs[0], 100)
    mc = MC_ElMap(traj_stacked, n=100, weighting='auto', spatial=0.1, shape=100.0, tangent=0.1, stretch=10.0, bend=10.0, init_nodes=init)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
    
    for i in range(len(cst_inds)):
        for j in range(np.shape(traj_stacked)[1]):
            print( abs(mc.solx.value[cst_inds[i] + (mc.n_nodes * j)][0] - cst_pts[i][j]) )
            
    plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k')
    plt.plot(repro[:, 0], repro[:, 1], 'r')
    for i in range(len(cst_inds)):
        plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
        plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    plt.show()

if __name__ == '__main__':
    main_lasa()