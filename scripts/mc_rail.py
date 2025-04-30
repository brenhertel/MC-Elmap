import numpy as np
import matplotlib.pyplot as plt

from mc_elmap import MC_elmap
from utils import *

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 18})

def rail_pressing():
    trajs = []
    traj_idx = 1
    users = [1, 2, 3, 4, 5, 6]
    for u in users:
        traj = read_RAIL_demo('PRESSING', u, traj_idx)
        trajs.append(traj)
        
    repros = []
    for i in range(len(trajs)):
        cst_inds = [0, 99]
        cst_pts = [trajs[i][0, :], trajs[i][-1, :]]
        
        # auto weighting
        mc = MC_elmap(trajs, n=100, weighting='auto')
        x = mc.setup_problem()
        mc.set_constraints(inds=cst_inds, csts=cst_pts)
        repro = mc.solve_problem()
        repros.append(repro)
        np.savetxt('../pictures/rail/pressing/repro' + str(i) + '.txt', repro)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(trajs)):
        ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', label='Demo')
    for i in range(len(repros)):
        ax.plot3D(repros[i][:, 0], repros[i][:, 1], repros[i][:, 2], 'r', label='Repro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.legend()
    plt.show()

def rail_pushing():
    trajs = []
    traj_idx = 1
    users = [1, 2, 3, 4, 5, 6]
    for u in users:
        traj = read_RAIL_demo('PUSHING', u, traj_idx)
        trajs.append(traj)
        
    repros = []
    for i in range(len(trajs)):
        cst_inds = [0, 99]
        cst_pts = [trajs[i][0, :], trajs[i][-1, :]]
        
        # auto weighting
        mc = MC_elmap(trajs, n=100, weighting='auto')
        x = mc.setup_problem()
        mc.set_constraints(inds=cst_inds, csts=cst_pts)
        repro = mc.solve_problem()
        repros.append(repro)
        np.savetxt('../pictures/rail/pushing/repro' + str(i) + '.txt', repro)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(trajs)):
        ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', label='Demo')
    for i in range(len(repros)):
        ax.plot3D(repros[i][:, 0], repros[i][:, 1], repros[i][:, 2], 'r', label='Repro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.legend()
    plt.show()

def rail_reaching():
    trajs = []
    traj_idx = 1
    users = [1, 2, 3, 4, 5, 6]
    for u in users:
        traj = read_RAIL_demo('REACHING', u, traj_idx)
        trajs.append(traj)
        
    repros = []
    for i in range(len(trajs)):
        cst_inds = [0, 99]
        cst_pts = [trajs[i][0, :], trajs[i][-1, :]]
        
        # auto weighting
        mc = MC_elmap(trajs, n=100, weighting='auto')
        x = mc.setup_problem()
        mc.set_constraints(inds=cst_inds, csts=cst_pts)
        repro = mc.solve_problem()
        repros.append(repro)
        np.savetxt('../pictures/rail/reaching/repro' + str(i) + '.txt', repro)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(trajs)):
        ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', label='Demo')
    for i in range(len(repros)):
        ax.plot3D(repros[i][:, 0], repros[i][:, 1], repros[i][:, 2], 'r', label='Repro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.legend()
    plt.show()
  
if __name__ == '__main__':
    rail_reaching()
    rail_pushing()
    rail_pressing()