import numpy as np
import matplotlib.pyplot as plt

from mc_elmap_v2 import MC_elmap
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
    
    #plt.legend()
    plt.show()
    #mysavefig(fig, '../pictures/rail/pressing/result')

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
    
    #plt.legend()
    #plt.show()
    mysavefig(fig, '../pictures/rail/pushing/result')

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
    
    #plt.legend()
    #plt.show()
    mysavefig(fig, '../pictures/rail/reaching/result')
    
def plot_fig_reaching():
    trajs = []
    traj_idx = 1
    users = [1, 2, 3, 4, 5, 6]
    for u in users:
        traj = read_RAIL_demo('REACHING', u, traj_idx)
        trajs.append(traj)
        
    repros = []
    for i in range(len(trajs)):
        repro = np.loadtxt('../pictures/rail/reaching/repro' + str(i) + '.txt')
        repros.append(repro)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(trajs)):
        ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', lw=3, alpha=0.4, label='Demo')
    for i in range(len(repros)):
        ax.plot3D(repros[i][:, 0], repros[i][:, 1], repros[i][:, 2], 'r', lw=2.5, alpha=0.9, label='Repro')
        ax.plot3D(repros[i][0, 0], repros[i][0, 1], repros[i][0, 2], 'r.', alpha=0.9, ms=15, label='Repro')
        ax.plot3D(repros[i][-1, 0], repros[i][-1, 1], repros[i][-1, 2], 'r.', alpha=0.9, ms=15, label='Repro')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    ax.view_init(azim=-35, elev=25)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #plt.legend()
    #plt.show()
    mysavefig(fig, '../pictures/rail/reaching/img2')
    
def plot_fig_pushing():
    trajs = []
    traj_idx = 1
    users = [1, 2, 3, 4, 5, 6]
    for u in users:
        traj = read_RAIL_demo('PUSHING', u, traj_idx)
        trajs.append(traj)
        
    repros = []
    for i in range(len(trajs)):
        repro = np.loadtxt('../pictures/rail/pushing/repro' + str(i) + '.txt')
        repros.append(repro)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(trajs)):
        ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', lw=3, alpha=0.4, label='Demo')
    for i in range(len(repros)):
        ax.plot3D(repros[i][:, 0], repros[i][:, 1], repros[i][:, 2], 'r', lw=2.5, alpha=0.9, label='Repro')
        ax.plot3D(repros[i][0, 0], repros[i][0, 1], repros[i][0, 2], 'r.', alpha=0.9, ms=15, label='Repro')
        ax.plot3D(repros[i][-1, 0], repros[i][-1, 1], repros[i][-1, 2], 'r.', alpha=0.9, ms=15, label='Repro')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    ax.view_init(azim=70, elev=20)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #plt.legend()
    #plt.show()
    mysavefig(fig, '../pictures/rail/pushing/img2')
    
def plot_fig_pressing():
    trajs = []
    traj_idx = 1
    users = [1, 2, 3, 4, 5, 6]
    for u in users:
        traj = read_RAIL_demo('PRESSING', u, traj_idx)
        trajs.append(traj)
        
    repros = []
    for i in range(len(trajs)):
        repro = np.loadtxt('../pictures/rail/pressing/repro' + str(i) + '.txt')
        repros.append(repro)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(trajs)):
        ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', lw=3, alpha=0.4, label='Demo')
    for i in range(len(repros)):
        ax.plot3D(repros[i][:, 0], repros[i][:, 1], repros[i][:, 2], 'r', lw=2.5, alpha=0.9, label='Repro')
        ax.plot3D(repros[i][0, 0], repros[i][0, 1], repros[i][0, 2], 'r.', alpha=0.9, ms=15, label='Repro')
        ax.plot3D(repros[i][-1, 0], repros[i][-1, 1], repros[i][-1, 2], 'r.', alpha=0.9, ms=15, label='Repro')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    ax.view_init(azim=-10, elev=25)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #plt.legend()
    #plt.show()
    mysavefig(fig, '../pictures/rail/pressing/img2')
    
if __name__ == '__main__':
    #rail_reaching()
    #rail_pushing()
    #rail_pressing()
    plot_fig_pressing()
    plot_fig_pushing()
    plot_fig_reaching()