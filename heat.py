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
#finite difference derivatives
def d2x(D):
    Dxx=np.zeros((ny,nx), dtype='double')
    for j in range(0,ny):
        for i in range(0,nx):
                 if(i==0):
                     Dxx[j][0]=(D[j][2]-2*D[j][1]+D[j][0])/(dx**2)
                 if(i==nx-1):
                     Dxx[j][nx-1]=(D[j][nx-1]-2*D[j][nx-1-1]+D[j][nx-1-2])/(dx**2)
                 if(i!=0 and i!=nx-1):
                     Dxx[j][i]=(D[j][i+1]-2*D[j][i]+D[j][i-1])/(dx**2)
    return Dxx

def d2y(D):
    Dyy=np.zeros((ny,nx), dtype='double')
    for j in range(0,ny):
        for i in range(0,nx):
                 if(j==0):
                     Dyy[j][0]=(D[2][i]-2*D[1][i]+D[0][i])/(dy**2)
                 if(j==ny-1):
                     Dyy[ny-1][i]=(D[ny-1][i]-2*D[ny-1-1][i]+D[ny-1-2][i])/(dy**2)
                 if(j!=0 and j!=ny-1):
                     Dyy[j][i]=(D[j+1][i]-2*D[j][i]+D[j-1][i])/(dy**2)
    return Dyy
