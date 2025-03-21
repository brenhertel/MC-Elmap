import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 18})
from utils import *

frechs = np.loadtxt('lasa_frechs.txt')
sses = np.loadtxt('lasa_sses.txt')
angs = np.loadtxt('lasa_angs.txt')
jerks = np.loadtxt('lasa_jerks.txt')

#25 shapes x [mc_elmap, uniform, cart, tan, shape, mccb] 

mce_frechs = frechs[:, 0] / np.amax(frechs)
uni_frechs = frechs[:, 1] / np.amax(frechs)
cart_frechs = frechs[:, 2] / np.amax(frechs)
tan_frechs = frechs[:, 3] / np.amax(frechs)
lap_frechs = frechs[:, 4] / np.amax(frechs)
mccb_frechs = frechs[:, 5] / np.amax(frechs)

fig = plt.figure()
ax = fig.add_subplot()
sns.boxplot(dict(zip(['Cart', 'Uni', 'MC-E', 'MCCB'], [cart_frechs, uni_frechs, mce_frechs, mccb_frechs])), boxprops={"facecolor": (.3, .5, .7, .5), "linewidth": 2}, medianprops={"color": "r", "linewidth": 1.5}, whiskerprops={"color": "k", "linewidth": 2}, capprops={"color": "k", "linewidth": 2}, meanline=True, meanprops={"color": "c", "linestyle" : "--", "linewidth": 1.5}, showmeans=False, ax=ax)
ax.set_yticks([])
#ax.set_xticklabels(['MC-Elmap', 'Uni', 'Cart', 'Tan', 'Lap', 'MCCB'], rotation = 30)
#plt.show()
mysavefig(fig, 'frechet_compare')

mce_sses = sses[:, 0] / np.amax(sses)
uni_sses = sses[:, 1] / np.amax(sses)
cart_sses = sses[:, 2] / np.amax(sses)
tan_sses = sses[:, 3] / np.amax(sses)
lap_sses = sses[:, 4] / np.amax(sses)
mccb_sses = sses[:, 5] / np.amax(sses)

fig = plt.figure()
ax = fig.add_subplot()
sns.boxplot(dict(zip(['Cart', 'Uni', 'MC-E', 'MCCB'], [cart_sses, uni_sses, mce_sses, mccb_sses])), boxprops={"facecolor": (.3, .5, .7, .5), "linewidth": 2}, medianprops={"color": "r", "linewidth": 1.5}, whiskerprops={"color": "k", "linewidth": 2}, capprops={"color": "k", "linewidth": 2}, meanline=True, meanprops={"color": "c", "linestyle" : "--", "linewidth": 1.5}, showmeans=False, ax=ax)
ax.set_yticks([])
#ax.set_xticklabels(['MC-Elmap', 'Uni', 'Cart', 'Tan', 'Lap', 'MCCB'], rotation = 30)
#plt.show()
mysavefig(fig, 'sse_compare')

mce_angs = angs[:, 0] / np.amax(angs)
uni_angs = angs[:, 1] / np.amax(angs)
cart_angs = angs[:, 2] / np.amax(angs)
tan_angs = angs[:, 3] / np.amax(angs)
lap_angs = angs[:, 4] / np.amax(angs)
mccb_angs = angs[:, 5] / np.amax(angs)

fig = plt.figure()
ax = fig.add_subplot()
sns.boxplot(dict(zip(['Cart', 'Uni', 'MC-E', 'MCCB'], [cart_angs, uni_angs, mce_angs, mccb_angs])), boxprops={"facecolor": (.3, .5, .7, .5), "linewidth": 2}, medianprops={"color": "r", "linewidth": 1.5}, whiskerprops={"color": "k", "linewidth": 2}, capprops={"color": "k", "linewidth": 2}, meanline=True, meanprops={"color": "c", "linestyle" : "--", "linewidth": 1.5}, showmeans=False, ax=ax)
ax.set_yticks([])
#ax.set_xticklabels(['MC-Elmap', 'Uni', 'Cart', 'Tan', 'Lap', 'MCCB'], rotation = 30)
#plt.show()
mysavefig(fig, 'ang_compare')

mce_jerks = jerks[:, 0] / np.amax(jerks)
uni_jerks = jerks[:, 1] / np.amax(jerks)
cart_jerks = jerks[:, 2] / np.amax(jerks)
tan_jerks = jerks[:, 3] / np.amax(jerks)
lap_jerks = jerks[:, 4] / np.amax(jerks)
mccb_jerks = jerks[:, 5] / np.amax(jerks)

fig = plt.figure()
ax = fig.add_subplot()
sns.boxplot(dict(zip(['Cart', 'Uni', 'MC-E', 'MCCB'], [cart_jerks, uni_jerks, mce_jerks, mccb_jerks])), boxprops={"facecolor": (.3, .5, .7, .5), "linewidth": 2}, medianprops={"color": "r", "linewidth": 1.5}, whiskerprops={"color": "k", "linewidth": 2}, capprops={"color": "k", "linewidth": 2}, meanline=True, meanprops={"color": "c", "linestyle" : "--", "linewidth": 1.5}, showmeans=False, ax=ax)
ax.set_yticks([])
#ax.set_xticklabels(['MC-Elmap', 'Uni', 'Cart', 'Tan', 'Lap', 'MCCB'], rotation = 30)
#plt.show()
mysavefig(fig, 'jerk_compare')