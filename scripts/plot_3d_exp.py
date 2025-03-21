import numpy as np
import matplotlib.pyplot as plt

from utils import *

trajs = []
for i in range(4):
    traj = np.loadtxt('traj_ds' + str(i+1) + '.txt')
    trajs.append(traj)
    
repro = np.loadtxt('mc_elmap_repro.txt')

fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(len(trajs)):
    ax.plot3D(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], 'k', lw=3, alpha=0.4, label='Demo')

ax.plot3D(repro[:, 0], repro[:, 1], repro[:, 2], 'r', lw=3, alpha=0.9, label='Repro')

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
ax.view_init(azim=-85, elev=50)
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
#plt.legend()
#plt.show()
mysavefig(fig, '../pictures/real/writing')