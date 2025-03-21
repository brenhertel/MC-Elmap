import numpy as np
import matplotlib.pyplot as plt

from utils import *
from downsampling import *
from mc_elmap_v2 import MC_elmap
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 16})

import similaritymeasures

def calc_sim(traj1, traj2):
    frechet = similaritymeasures.frechet_dist(traj1, traj2)
    sse = align_SSE(traj1, traj2)
    ang = align_ang_sim(traj1, traj2)
    jerk = calc_jerk(traj2)
    return [frechet, sse, ang, jerk]


lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
                'heee', 'JShape','Khamesh','Leaf_1', \
                'Leaf_2','Line','LShape','NShape','PShape', \
                'RShape','Saeghe','Sharpc','Sine', 'Snake', \
                'Spoon','Sshape','Trapezoid','Worm','WShape', 'Zshape']

def main_lasa_1d(shape_name):
    trajs = []
    starts = []
    for i in range(1, 2):
        trajs.append(get_lasa_trajN(shape_name, n=i)[:, 0].reshape((1000, 1)))
        starts.append(trajs[i-1][0])
    mean_start = np.mean(np.vstack(starts), axis=0)
    
    cst_inds = [0, len(trajs[0])-1]
    cst_pts = [mean_start, trajs[0][-1]]
    init = DouglasPeuckerPoints(trajs[0], 100)
        
    # auto weighting
    print(shape_name + " AUTO")
    mc = MC_ElMap(trajs, n=1000, weighting='auto', ds=False)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i], 'k', lw=3, alpha=0.4)
    plt.plot(repro, 'r', lw=4)
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], repro[cst_inds[i]], 'r.', ms=15)
    mysavefig(fig, '../pictures/lasa/auto_weighting/' + shape_name)
    np.savetxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt', repro)
        
    # uniform weighting
    print(shape_name + " UNIFORM")
    mc = MC_ElMap(trajs, n=1000, weighting='half-auto', spatial=0.33, shape=0.33, tangent=0.33, stretch=0.01, bend=0.01, ds=False)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i], 'k', lw=3, alpha=0.4)
    plt.plot(repro, 'r', lw=4)
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], repro[cst_inds[i]], 'r.', ms=15)
    mysavefig(fig, '../pictures/lasa/uniform_weighting/' + shape_name)
    np.savetxt('../pictures/lasa/uniform_weighting/' + shape_name + '.txt', repro)
        
    # cart only
    print(shape_name + " CART")
    mc = MC_ElMap(trajs, n=1000, weighting='half-auto', spatial=1.0, shape=0.0, tangent=0.0, stretch=0.01, bend=0.01, ds=False)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i], 'k', lw=3, alpha=0.4)
    plt.plot(repro, 'r', lw=4)
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], repro[cst_inds[i]], 'r.', ms=15)
    mysavefig(fig, '../pictures/lasa/cart_only/' + shape_name)
    np.savetxt('../pictures/lasa/cart_only/' + shape_name + '.txt', repro)
        
    # tangent only
    print(shape_name + " TAN")
    mc = MC_ElMap(trajs, n=1000, weighting='half-auto', spatial=0.0, shape=0.0, tangent=1.0, stretch=0.01, bend=0.01, ds=False)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i], 'k', lw=3, alpha=0.4)
    plt.plot(repro, 'r', lw=4)
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], repro[cst_inds[i]], 'r.', ms=15)
    mysavefig(fig, '../pictures/lasa/tangent_only/' + shape_name)
    np.savetxt('../pictures/lasa/tangent_only/' + shape_name + '.txt', repro)
        
    # cart only
    print(shape_name + " SHAPE")
    mc = MC_ElMap(trajs, n=1000, weighting='half-auto', spatial=0.0, shape=1.0, tangent=0.0, stretch=0.01, bend=0.01, ds=False)
    x = mc.setup_problem()
    mc.set_constraints(inds=cst_inds, csts=cst_pts)
    repro = mc.solve_problem()
            
    fig = plt.figure()
    for i in range(len(trajs)):
        plt.plot(trajs[i], 'k', lw=3, alpha=0.4)
    plt.plot(repro, 'r', lw=4)
    for i in range(len(cst_inds)):
        plt.plot(cst_inds[i], repro[cst_inds[i]], 'r.', ms=15)
    mysavefig(fig, '../pictures/lasa/shape_only/' + shape_name)
    np.savetxt('../pictures/lasa/shape_only/' + shape_name + '.txt', repro)

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
            
    #fig = plt.figure()
    #for i in range(len(trajs)):
    #    plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    #plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    #for i in range(len(cst_inds)):
    #    #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
    #    plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    #mysavefig(fig, '../pictures/lasa/auto_weighting/' + shape_name)
    #np.savetxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt', repro)
    #    
    ## uniform weighting
    #print(shape_name + " UNIFORM")
    #mc = MC_elmap(trajs, n=100, weighting='half-auto', spatial=0.33, shape=0.33, tangent=0.33, stretch=1.0, bend=1.0)
    #x = mc.setup_problem()
    #mc.set_constraints(inds=cst_inds, csts=cst_pts)
    #repro = mc.solve_problem()
    #        
    #fig = plt.figure()
    #for i in range(len(trajs)):
    #    plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    #plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    #for i in range(len(cst_inds)):
    #    #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
    #    plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    #mysavefig(fig, '../pictures/lasa/uniform_weighting/' + shape_name)
    #np.savetxt('../pictures/lasa/uniform_weighting/' + shape_name + '.txt', repro)
    #    
    ## cart only
    #print(shape_name + " CART")
    #mc = MC_elmap(trajs, n=100, weighting='half-auto', spatial=1.0, shape=0.0, tangent=0.0, stretch=1.0, bend=1.0)
    #x = mc.setup_problem()
    #mc.set_constraints(inds=cst_inds, csts=cst_pts)
    #repro = mc.solve_problem()
    #        
    #fig = plt.figure()
    #for i in range(len(trajs)):
    #    plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    #plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    #for i in range(len(cst_inds)):
    #    #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
    #    plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    #mysavefig(fig, '../pictures/lasa/cart_only/' + shape_name)
    #np.savetxt('../pictures/lasa/cart_only/' + shape_name + '.txt', repro)
    #    
    ## tangent only
    #print(shape_name + " TAN")
    #mc = MC_elmap(trajs, n=100, weighting='half-auto', spatial=0.0, shape=0.0, tangent=1.0, stretch=0.01, bend=0.01)
    #x = mc.setup_problem()
    #mc.set_constraints(inds=cst_inds, csts=cst_pts)
    #repro = mc.solve_problem()
    #        
    #fig = plt.figure()
    #for i in range(len(trajs)):
    #    plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    #plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    #for i in range(len(cst_inds)):
    #    #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
    #    plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    #mysavefig(fig, '../pictures/lasa/tangent_only/' + shape_name)
    #np.savetxt('../pictures/lasa/tangent_only/' + shape_name + '.txt', repro)
    #    
    ## cart only
    #print(shape_name + " SHAPE")
    #mc = MC_elmap(trajs, n=100, weighting='half-auto', spatial=0.0, shape=1.0, tangent=0.0, stretch=0.01, bend=0.01)
    #x = mc.setup_problem()
    #mc.set_constraints(inds=cst_inds, csts=cst_pts)
    #repro = mc.solve_problem()
    #        
    #fig = plt.figure()
    #for i in range(len(trajs)):
    #    plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    #plt.plot(repro[:, 0], repro[:, 1], 'r', lw=4)
    #for i in range(len(cst_inds)):
    #    #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
    #    plt.plot(repro[cst_inds[i], 0], repro[cst_inds[i], 1], 'r.', ms=15)
    #mysavefig(fig, '../pictures/lasa/shape_only/' + shape_name)
    #np.savetxt('../pictures/lasa/shape_only/' + shape_name + '.txt', repro)

def main_compare():
    frechs = np.zeros((25, 6))
    sses = np.zeros((25, 6))
    angs = np.zeros((25, 6))
    jerks = np.zeros((25, 6))
    for n in range(len(lasa_names)):
    
        shape_name = lasa_names[n]
        
        trajs = []
        starts = []
        for i in range(1, 8):
            trajs.append(get_lasa_trajN(shape_name, n=i))
            starts.append(trajs[i-1][0, :])
        mean_start = np.mean(np.vstack(starts), axis=0)
        
        repro_mcelmap = np.loadtxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt')
        
        r_frechs = []
        r_sses = []
        r_angs = []
        r_jerks = []
        for i in range(len(trajs)):
            [f, s, a, j] = calc_sim(trajs[i], repro_mcelmap)
            r_frechs.append(f)
            r_sses.append(s)
            r_angs.append(a)
            r_jerks.append(j)
        frechs[n, 0] = np.mean(r_frechs)
        sses[n, 0] = np.mean(r_sses)
        angs[n, 0] = np.mean(r_angs)
        jerks[n, 0] = np.mean(r_jerks)
        
        repro_uniform = np.loadtxt('../pictures/lasa/uniform_weighting/' + shape_name + '.txt')
        
        r_frechs = []
        r_sses = []
        r_angs = []
        r_jerks = []
        for i in range(len(trajs)):
            [f, s, a, j] = calc_sim(trajs[i], repro_uniform)
            r_frechs.append(f)
            r_sses.append(s)
            r_angs.append(a)
            r_jerks.append(j)
        frechs[n, 1] = np.mean(r_frechs)
        sses[n, 1] = np.mean(r_sses)
        angs[n, 1] = np.mean(r_angs)
        jerks[n, 1] = np.mean(r_jerks)
        
        repro_cart = np.loadtxt('../pictures/lasa/cart_only/' + shape_name + '.txt')
        
        r_frechs = []
        r_sses = []
        r_angs = []
        r_jerks = []
        for i in range(len(trajs)):
            [f, s, a, j] = calc_sim(trajs[i], repro_cart)
            r_frechs.append(f)
            r_sses.append(s)
            r_angs.append(a)
            r_jerks.append(j)
        frechs[n, 2] = np.mean(r_frechs)
        sses[n, 2] = np.mean(r_sses)
        angs[n, 2] = np.mean(r_angs)
        jerks[n, 2] = np.mean(r_jerks)
        
        repro_tan = np.loadtxt('../pictures/lasa/tangent_only/' + shape_name + '.txt')
        
        r_frechs = []
        r_sses = []
        r_angs = []
        r_jerks = []
        for i in range(len(trajs)):
            [f, s, a, j] = calc_sim(trajs[i], repro_tan)
            r_frechs.append(f)
            r_sses.append(s)
            r_angs.append(a)
            r_jerks.append(j)
        frechs[n, 3] = np.mean(r_frechs)
        sses[n, 3] = np.mean(r_sses)
        angs[n, 3] = np.mean(r_angs)
        jerks[n, 3] = np.mean(r_jerks)
        
        repro_shape = np.loadtxt('../pictures/lasa/shape_only/' + shape_name + '.txt')
        
        r_frechs = []
        r_sses = []
        r_angs = []
        r_jerks = []
        for i in range(len(trajs)):
            [f, s, a, j] = calc_sim(trajs[i], repro_shape)
            r_frechs.append(f)
            r_sses.append(s)
            r_angs.append(a)
            r_jerks.append(j)
        frechs[n, 4] = np.mean(r_frechs)
        sses[n, 4] = np.mean(r_sses)
        angs[n, 4] = np.mean(r_angs)
        jerks[n, 4] = np.mean(r_jerks)
        
        repro_mccb = np.loadtxt('../pictures/lasa/mccb/' + shape_name + '_sol.txt')
        
        r_frechs = []
        r_sses = []
        r_angs = []
        r_jerks = []
        for i in range(len(trajs)):
            [f, s, a, j] = calc_sim(trajs[i], repro_mccb)
            r_frechs.append(f)
            r_sses.append(s)
            r_angs.append(a)
            r_jerks.append(j)
        frechs[n, 5] = np.mean(r_frechs)
        sses[n, 5] = np.mean(r_sses)
        angs[n, 5] = np.mean(r_angs)
        jerks[n, 5] = np.mean(r_jerks)

        print(frechs)
        print(sses)
        print(angs)
        print(jerks)
        
    np.savetxt('lasa_frechs.txt', frechs)
    np.savetxt('lasa_sses.txt', sses)
    np.savetxt('lasa_angs.txt', angs)
    np.savetxt('lasa_jerks.txt', jerks)

def main_replot():
    for n in range(len(lasa_names)):
    
        shape_name = lasa_names[n]
        
        trajs = []
        starts = []
        for i in range(1, 8):
            trajs.append(get_lasa_trajN(shape_name, n=i))
            starts.append(trajs[i-1][0, :])
        mean_start = np.mean(np.vstack(starts), axis=0)
        cst_inds = [0, 99]
        cst_pts = [mean_start, trajs[0][-1, :]]
        
        repro_mcelmap = np.loadtxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt')
    
        fig = plt.figure()
        for i in range(len(trajs)):
            plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
        plt.plot(repro_mcelmap[:, 0], repro_mcelmap[:, 1], 'r', lw=5, alpha=0.9)
        for i in range(len(cst_inds)):
            #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
            plt.plot(repro_mcelmap[cst_inds[i], 0], repro_mcelmap[cst_inds[i], 1], 'r.', ms=15)
        plt.xticks([])
        plt.yticks([])
        mysavefig(fig, '../pictures/lasa/auto_weighting/' + shape_name)
        #np.savetxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt', repro)
        
def nshape_replot():
    
    shape_name = 'NShape'
    
    trajs = []
    starts = []
    for i in range(1, 8):
        trajs.append(get_lasa_trajN(shape_name, n=i))
        starts.append(trajs[i-1][0, :])
    mean_start = np.mean(np.vstack(starts), axis=0)
    cst_inds = [0, 99]
    cst_pts = [mean_start, trajs[0][-1, :]]
    
    repro_mcelmap = np.loadtxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt')
    repro_mcuni = np.loadtxt('../pictures/lasa/uniform_weighting/' + shape_name + '.txt')
    
    fig = plt.figure()
    plt.plot(trajs[0][:, 0], trajs[0][:, 1], 'k', lw=3, alpha=0.4, label='Demos')
    for i in range(1, len(trajs)):
        plt.plot(trajs[i][:, 0], trajs[i][:, 1], 'k', lw=3, alpha=0.4)
    plt.plot(repro_mcelmap[:, 0], repro_mcuni[:, 1], 'g', lw=5, alpha=0.9, label='Uniform')
    plt.plot(repro_mcelmap[:, 0], repro_mcelmap[:, 1], 'r', lw=5, alpha=0.9, label='MC-Elmap')
    for i in range(len(cst_inds)):
        #plt.plot(cst_pts[i][0], cst_pts[i][1], 'k.', ms=15)
        plt.plot(repro_mcelmap[cst_inds[i], 0], repro_mcelmap[cst_inds[i], 1], 'r.', ms=15)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right', bbox_to_anchor=(0.9, 1.0))
    mysavefig(fig, '../pictures/lasa/' + shape_name)
    #np.savetxt('../pictures/lasa/auto_weighting/' + shape_name + '.txt', repro)
    #plt.show()

if __name__ == '__main__':
    #for name in lasa_names:
    #    main_lasa(name)
    #main_compare()
    #main_replot()
    nshape_replot()