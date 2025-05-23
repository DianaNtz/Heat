"""Runge Kutta method for the 2D heat equation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import imageio.v2 as imageio
nfilenames = []
afilenames = []
#some initial values
nx=30
x0=0
xfinal=1
dx=(xfinal-x0)/(nx-1)
x=np.linspace(x0,xfinal,nx)

ny=30
y0=0
yfinal=1
dy=(yfinal-y0)/(ny-1)
y=np.linspace(y0,yfinal,ny)

nt=450
t0=0
tfinal=0.17
dt=(tfinal-t0)/(nt-1)
t=0
