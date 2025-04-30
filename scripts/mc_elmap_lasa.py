import numpy as np
import matplotlib.pyplot as plt

from utils import *
from downsampling import *
from mc_elmap import MC_elmap
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 16})


lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
                'heee', 'JShape','Khamesh','Leaf_1', \
                'Leaf_2','Line','LShape','NShape','PShape', \
                'RShape','Saeghe','Sharpc','Sine', 'Snake', \
                'Spoon','Sshape','Trapezoid','Worm','WShape', 'Zshape']

def main_lasa(shape_name):
    trajs = []
    starts = []
    for i in range(1, 8):
        trajs.append(get_lasa_trajN(shape_name, n=i))
        starts.append(trajs[i-1][0, :])
    mean_start = np.mean(np.vstack(starts), axis=0)
    
    cst_inds = [0, 99]
    cst_pts = [mean_start, trajs[0][-1, :]]
    init = DouglasPeuckerPoints(trajs[0], 100)
        
    # auto weighting
    print(shape_name + " AUTO")
    mc = MC_elmap(trajs, n=100, weighting='auto')
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
    print(shape_name + ' ALPHAS: ' + str([mc.spatial_alpha, mc.tangent_alpha, mc.shape_alpha]))
            
    fig = plt.figure()
    plt.title('Auto')
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    for i in range(len(cst_inds)):
        #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
        plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
        
    # uniform weighting
    print(shape_name + " UNIFORM")
    mc = MC_elmap(trajs, n=100, weighting='half-auto', spatial=0.33, shape=0.33, tangent=0.33, stretch=1.0, bend=1.0)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    plt.title('Uniform')
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    for i in range(len(cst_inds)):
        #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
        plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
        
    # cart only
    print(shape_name + " CART")
    mc = MC_elmap(trajs, n=100, weighting='half-auto', spatial=1.0, shape=0.0, tangent=0.0, stretch=1.0, bend=1.0)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    plt.title('Cart')
    for i in range(len(trajs)):
        plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    for i in range(len(cst_inds)):
        #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
        plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
        
    plt.show()


if __name__ == '__main__':
    for name in lasa_names:
        main_lasa(name)