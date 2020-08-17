# -*- coding: utf-8 -*-

"""
CALCULATION OF THE CRITICAL LAND USE STRATEGY VALUES FOR DIFFERENT VALUES OF THE
LAND RECOVERY AND DEGRADATION PARAMETERS (2 CONTOUR PLOTS).
CALCULATION OF THE SIZE OF THE COLLAPSE REGION AS A FUNCTION OF THE LAND RECOVERY
RATE.

THE CODE CARRIES A BIFURCATION ANALYSIS TO IDENTIFY THE CRITICAL VALUES OF b

model variables:
pop=human population density
l0=natural Land
l1=degraded land

model parameters:

1) b=land use strategy (0->extensive agriculture; 1->intensive agriculture)
2) K=maximum land conversion effort
3) Kmin=minimum land conversion effort
NB: actual conversion effort = Kmin+(K-Kmin)*(1-b)

4) R=recovery rate of degraded land
5) D=degradation rate of natural land

6) E=maximum degradation rate of agricultural land
NB: actual degradation rate = E*b

7) Q=relative importance of natural land to agricultural production
"""


import numpy as np
from scipy import optimize # LIBRARY NEEDED FOR ROOT FINDING
from scipy import linalg # LIBRARY NEEDED FOR EIGENVALUES
import os
from matplotlib import pyplot as plt
import time
import pylab as pl
from matplotlib import cm
import matplotlib.colors
from scipy import ndimage, misc
from scipy import interpolate
import math
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
#import seaborn as sns
import matplotlib.style as style
import seaborn as sns

clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
               param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                           "gamma": np.logspace(-2, 2, 5)})


t0 = time.time()

#plt.style.use('seaborn')

"""
FUNCTION DECLARATION
"""
def foodProduction(state,param):
    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    effective_land=(l0+l1*(1-b))
    a=1-l0-l1
    prod=b*(b*a+Q*(1-b)*effective_land*np.power(a,0.5))

    return prod

def popEq(state,param):

    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    prod=foodProduction(state,param)

    return pop*(1-pop/prod)

def l0Eq(state,param):

    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    if l0<0:
        l0=0
    if l1<0:
        l1=0

    ret = (R*np.power(l0,0.5)-b*D*np.power(l1,0.5))*np.power(l1*l0,0.5)-(Kmin+(K-Kmin)*(1-b))*pop*l0
    return ret

def l1Eq(state,param):

    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    if l0<0:
        l0=0
    if l1<0:
        l1=0

    a=1-l0-l1
    ret= -(R*np.power(l0,0.5)-b*D*np.power(l1,0.5))*np.power(l1*l0,0.5)+b*E*a
    return ret

def fun(state,param):

    return [popEq(state,param), l0Eq(state,param), l1Eq(state,param)]

def analytical_jacobian(state,param):

    pop=state[0];l0=state[1];l1=state[2]
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];F=param[5];Kmin=param[6]

    jac=np.zeros((3,3))

    effective_land=(l0+l1*(1-b))
    a=1-l0-l1
    prod=b*(b*a+F*(1-b)*effective_land*np.power(a,0.5))

    derivl0prod=-b*b + F*b*(1-b)*( np.power(a,0.5) - 0.5*effective_land/np.power(a,0.5) )
    derivl1prod=-b*b + F*b*(1-b)*( (1-b)*np.power(a,0.5) - 0.5*effective_land/np.power(a,0.5) )

    jac[0,0]=1-2*pop/prod
    jac[0,1]=-pop*pop*derivl0prod/prod/prod
    jac[0,2]=-pop*pop*derivl1prod/prod/prod

    jac[1,0]=-K*(1-b)*l0;
    jac[1,1]=R*np.sqrt(l1)-b*D*l1*0.5*np.power(l0,-0.5)-K*(1-b)*pop;
    jac[1,2]=R*l0*0.5*np.power(l1,-0.5)-b*D*np.sqrt(l0)

    jac[2,0]=0
    jac[2,1]=-(R*np.sqrt(l1)-b*D*l1*0.5*np.power(l0,-0.5))-b*E
    jac[2,2]=-(R*l0*0.5*np.power(l1,-0.5)-b*D*np.sqrt(l0))-b*E

    ret = jac
    return ret

def jacobian(state,param):

    pop=state[0];l0=state[1];l1=state[2]
    dstate=[1e-6,1e-6,1e-6]
    dp=dstate[0];dl0=dstate[1];dl1=dstate[2]

    # states with the added perturbation
    pstate=[pop+dp,l0,l1]
    l0state=[pop,l0+dl0,l1]
    l1state=[pop,l0,l1+dl1]

    pert_state=[pstate,l0state,l1state]

    Jac=np.zeros((3,3))
    for i in range(3):
        Jac[0,i]=(popEq(pert_state[i],param)-popEq(state,param))/dstate[i]
        Jac[1,i]=(l0Eq(pert_state[i],param)-l0Eq(state,param))/dstate[i]
        Jac[2,i]=(l1Eq(pert_state[i],param)-l1Eq(state,param))/dstate[i]

    ret = Jac
    return ret


def solver_RK4(state,param,dt):

    k1p=dt*popEq(state,param);k1l0=dt*l0Eq(state,param);k1l1=dt*l1Eq(state,param);
    k1vec=np.array([k1p,k1l0,k1l1])
    state1=state+0.5*k1vec

    k2p=dt*popEq(state1,param);k2l0=dt*l0Eq(state1,param);k2l1=dt*l1Eq(state1,param);
    k2vec=np.array([k2p,k2l0,k2l1])
    state2=state+0.5*k2vec

    k3p=dt*popEq(state2,param);k3l0=dt*l0Eq(state2,param);k3l1=dt*l1Eq(state2,param);
    k3vec=np.array([k3p,k3l0,k3l1])
    state3=state+k3vec

    k4p=dt*popEq(state3,param);k4l0=dt*l0Eq(state3,param);k4l1=dt*l1Eq(state3,param);
    k4vec=np.array([k4p,k4l0,k4l1])

    var_state=np.array((k1vec+2*k2vec+2*k3vec+k4vec)/6)

    return state+var_state

def initial_guess(param):

    T=200;dt=0.001;t=0
    state=np.array([0.1,0.8,0.1])
    state_new=np.zeros(3)

    t_show=0
    while t<T:
        state_new=solver_RK4(state,param,dt)
        deriv=(state_new-state)/dt
        if np.sqrt(np.dot(deriv,deriv))<1e-6:# steadystate reached
            break
        state=state_new

        if t>t_show:
            print(state_new)
            t_show+=10

        t=t+dt

    return state

def savearray(var,par,par_arr,array):

    line=np.where(par_arr[0]==par[0]);col=np.where(par_arr[1]==par[1])
    array[line,col]=var
    #print(array)
    return array

def plot(data,par_arr):
    #print(data)

    Rarray=np.array(par_arr[0])
    Darray=np.array(par_arr[1])

    D,R=np.meshgrid(Darray,Rarray)

    fig, (ax) = plt.subplots(1)
    origin='lower'

    # good=(~np.isnan(data))
    # print(good)
    #
    # XY=np.array([R[good],D[good]]).T
    # print(XY.shape)
    #
    # clf.fit(XY,data[good])
    #
    # Rmin=0.1;Rmax=2.0;Rdatapoints=100
    # Rsmooth=np.linspace(Rmin,Rmax,Rdatapoints)
    # Dmin=0.1;Dmax=2.0;Ddatapoints=100
    # Dsmooth=np.linspace(Dmin,Dmax,Ddatapoints)
    #
    # R2,D2=np.meshgrid(Rsmooth,Dsmooth)
    #
    #
    # XYall=np.array([R2.ravel(),D2.ravel()]).T
    # Zall = clf.predict(XYall)
    #
    # Z=Zall.reshape(R2.shape)
    #
    # ind=np.where(Z>0.45)
    # Z[ind]=np.nan

    #fig, (ax1,ax2)=plt.subplots(2)

    plt.gca().patch.set_color('.15')
    #cp=ax1.contourf(R2,D2,Z,origin=origin,alpha=0.9)
    cp=ax.contourf(R,D,data,origin=origin,alpha=0.9)

    #norm=matplotlib.colors.Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
    #sm = plt.cm.ScalarMappable(norm=norm, cmap = cp.cmap)
    #sm.set_array([])
    fig.colorbar(cp,ax=ax,ticks=cp.levels)
    #fig.colorbar(cp2,ax=ax2,ticks=cp.levels)

    ax.set_title(r'$\beta_c$')
    ax.set_xlabel('R')
    ax.set_ylabel('D')

    plt.show()

def plotful(data1,data2,dataarraydiff,par_arr):
    #sns.set_style("ticks")
    sns.set_context('paper')
    style.use('seaborn-paper')

    Rarray=np.array(par_arr[0])
    Darray=np.array(par_arr[1])
    D,R=np.meshgrid(Darray,Rarray)

    fig = plt.figure()
    # ax1 = plt.subplot(221)
    # ax2 = plt.subplot(223)
    # ax3 = plt.subplot(122)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)

    origin='lower'
    ax1.set_facecolor('0.2')
    ax2.set_facecolor('0.2')

    cp1=ax1.contourf(R,D,data1,origin=origin,alpha=0.9)
    cp2=ax2.contourf(R,D,data2,origin=origin,alpha=0.9)

    deltaD=Darray[1]-Darray[0]
    iD1=np.where(Darray>0.5-0.01)[0][0]
    iD2=np.where(Darray>1.0-0.01)[0][0]
    iD3=np.where(Darray>1.5-0.01)[0][0]

    datad1=dataarraydiff[:,iD1]
    datad2=dataarraydiff[:,iD2]
    datad3=dataarraydiff[:,iD3]

    # Rmin=0.5;Rmax=2.0;
    # Rnew = np.linspace(Rmin, Rmax, 200)
    # f = interpolate.interp1d(Rarray,datad1,kind='cubic')
    # datad1new = f(Rnew)   # use interpolation function returned by `interp1d`
    # f = interpolate.interp1d(Rarray,datad2,kind='cubic')
    # datad2new = f(Rnew)   # use interpolation function returned by `interp1d`
    # f = interpolate.interp1d(Rarray,datad3,kind='cubic')
    # datad3new = f(Rnew)   # use interpolation function returned by `interp1d`

    D1=np.around(Darray[iD1],2);D2=np.around(Darray[iD2],2);D3=np.around(Darray[iD3],3)

    ax3.plot(Rarray,datad1,'o',linewidth=2,alpha=0.8,label="d="+str(D1))
    ax3.plot(Rarray,datad2,'o',linewidth=2,alpha=0.8,label="d="+str(D2))
    ax3.plot(Rarray,datad3,'o',linewidth=2,alpha=0.8,label="d="+str(D3))
    # ax3.plot(Rnew,datad1new,linewidth=2,label="d="+str(D1),color='tab:blue',alpha=0.6)
    # ax3.plot(Rnew,datad2new,linewidth=2,label="d="+str(D2),color='tab:orange',alpha=0.6)
    # ax3.plot(Rnew,datad3new,linewidth=2,label="d="+str(D3),color='tab:green',alpha=0.6)

    #norm=matplotlib.colors.Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
    #sm = plt.cm.ScalarMappable(norm=norm, cmap = cp.cmap)
    #sm.set_array([])

    # pcm1 = ax1.pcolormesh(R, D, data1,cmap='viridis',alpha=0.9)
    # pcm2 = ax2.pcolormesh(R, D, data2,cmap='viridis',alpha=0.9)

    fig.colorbar(cp1,ax=ax1,ticks=cp1.levels)
    fig.colorbar(cp2,ax=ax2,ticks=cp2.levels)

    ax1.set_title(r'$\beta_{c,1}$')
    ax2.set_title(r'$\beta_{c,2}$')

    ax1.set_ylabel('d')
    ax2.set_ylabel('d')
    ax1.set_xlabel('r\n(a)')
    ax2.set_xlabel('r\n(b)')
    ax3.set_xlabel('r\n(c)')
    ax3.set_ylabel(r'$\Delta \beta$')
    #sns.despine()
    ax3.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    sns.despine()
    plt.show()


"""
PARAMETER & VARIABLE DECLARATION
"""

E=1
F=1
K=4.5
Kmin=0.5

initial_state=np.array([0.01,0.98,0.01])

dstate=np.ones(5)*1e-7

"""
critical b in R,D plane
"""

Rmin=0.5;Rmax=2.0;Rdatapoints=0.05 # it is actually deltaR
Rarray=np.arange(Rmin,Rmax,Rdatapoints)
Dmin=0.5;Dmax=2.0;Ddatapoints=0.05
Darray=np.arange(Dmin,Dmax,Ddatapoints)
par_arr=[Rarray,Darray]
dataarray=np.empty((len(Rarray),len(Darray),))
dataarray[:]=np.nan

bmin=0.02;bmax=0.95;db=0.001
b=bmin

param=[bmin,K,Rmin,Dmin,E,F,Kmin]
old_fpR=initial_guess(param)

print('entering to bc1')
# this is for betac1
for R in Rarray:
    old_fpD=old_fpR
    for D in Darray:
        old_fp=old_fpD
        b=bmin
        while b<bmax:

            param=[b,K,R,D,E,F,Kmin]
            #print("b = "+str(b))
            # continuate initial state
            state0=old_fp

            # solve the root problem and get the fixed point
            #sol = optimize.root(fun,state0,args=(param,),method='lm',jac=analytical_jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})
            sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})
            fixed_point = sol.x # steady states

            #print(state0)
            #print(fixed_point-state0)

            # test if the root finder actually converged to a fixed point
            btest=np.array(fun(fixed_point,param))
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test>1e-6):# if it didn't converge
                print("b = "+str(b)+"; R="+str(R)+"; D="+str(D))
                # this means we reached the critical point, save it and break loop
                # print("here")
                # print(l2test)
                # print(state0)
                # print(fixed_point)
                # print("end")
                bc=b
                par=[R,D]
                dataarray=savearray(bc,par,par_arr,dataarray)
                break

            # test if there was convergence to trivial state!
            tstate=[0.0,0.0,1.0]
            btest=fixed_point-tstate
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test<1e-6): # it converged to the trivial state
                print("b = "+str(b)+"; R="+str(R)+"; D="+str(D))
                # this means we reached the critical point, save it and break loop
                bc=b
                par=[R,D]
                dataarray=savearray(bc,par,par_arr,dataarray)
                break

            old_fp=np.array(fixed_point)

            if b<bmin+db:
                old_fpD=old_fp
                if D<Darray[1]:
                    old_fpR=old_fp

            b=b+db
# frontier=np.array([[0,0]])
# print(dataarray)
# for i in range(dataarray.shape[0]):
#     print(i)
#     for j in range(dataarray.shape[1]):
#         print(dataarray[i,j])
#         if math.isnan(dataarray[i,j]):
#             print("here")
#             frontier=np.concatenate((frontier,[[i,j]]))
#             break
#
# frontier=frontier[1:][:]
# print(frontier)

# # what about interpolating the data
#plot(dataarray,par_arr)

np.savetxt('betacrit1.dat',dataarray)
print('entering to bc2')

dataarray2=np.empty((len(Rarray),len(Darray),))
dataarray2[:]=np.nan
param=[bmax,K,Rmin,Dmin,E,F,Kmin]
# print("here")
old_fpR=initial_guess(param)
# print(old_fpR)
# this is for betac2
for R in Rarray:
    # print(R)
    old_fpD=old_fpR
    for D in Darray:
        old_fp=old_fpD
        b=bmax
        while b>bmin:
            # print("first: b = "+str(b)+"; R="+str(R)+"; D="+str(D))
            param=[b,K,R,D,E,F,Kmin]

            # calculate initial guess or keep preceeding
            state0=old_fp

            # solve the root problem and get the fixed point
            sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})
            fixed_point = sol.x # steady states

            # print(fixed_point)
            # test if the root finder actually converged to a fixed point
            btest=np.array(fun(fixed_point,param))
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test>1e-6):# if it didn't converge
                print("b = "+str(b)+"; R="+str(R)+"; D="+str(D))
                # print("here")
                #print(param[2:4])
                # this means we reached the critical point, save it and break loop
                bc=b
                par=[R,D]
                dataarray2=savearray(bc,par,par_arr,dataarray2)
                break

            # test if there was convergence to trivial state!
            tstate=[0.0,0.0,1.0]
            btest=fixed_point-tstate
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test<1e-6): # it converged to the trivial state
                print("b = "+str(b)+"; R="+str(R)+"; D="+str(D))
                # this means we reached the critical point, save it and break loop
                bc=b
                par=[R,D]
                dataarray2=savearray(bc,par,par_arr,dataarray2)
                break

            old_fp=np.array(fixed_point)

            if b>bmax-db:
                old_fpD=old_fp
                if D<Darray[1]:
                    old_fpR=old_fp


            b=b-db

np.savetxt('betacrit2.dat',dataarray2)

# what about interpolating the data
#plot(dataarray2,par_arr)
#
# # distance between bc1 and bc2
dataarraydiff=np.abs(dataarray2-dataarray)
change=np.isnan(dataarraydiff)
dataarraydiff[change]=0
np.savetxt('deltabeta.dat',dataarraydiff)

plotful(dataarray,dataarray2,dataarraydiff,par_arr)
plot(dataarraydiff,par_arr)
