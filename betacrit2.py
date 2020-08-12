# -*- coding: utf-8 -*-

"""
TEMPORAL DYANMICS PLOTS IN THE K,I PLANE
"""

import numpy as np
import math
from scipy import optimize # LIBRARY NEEDED QOR ROOT FINDING
from scipy import linalg # LIBRARY NEEDED FOR EIGENVALUES
import os
from matplotlib import pyplot as plt
import time
import pylab as pl
from matplotlib import cm
import matplotlib.colors
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.style as style

#plt.style.use('seaborn')

t0 = time.time()

"""
FUNCTION DECLARATION
"""

def foodProduction(state,param):
    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    d=1-a-n
    # if a<0:
    #     a=0
    # if d<0:
    #     d=0
    # if n<0:
    #     n=0

    effective_land=(n+d*(1-b))
    prod=b*(b*a+Q*(1-b)*effective_land*np.sqrt(a))

    return prod

###############################################################################
# dynamical equations

def pEq(state,param):

    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    prod=foodProduction(state,param)

    return p*(1-p/prod)

def nEq(state,param):

    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    d=1-a-n

    # if a<0:
    #     a=0
    # if d<0:
    #     d=0
    # if n<0:
    #     n=0

    # ret = (R*np.power(n,0.5)-b*D*np.power(d,0.5))*np.power(d*n,0.5)-(Kmin+(K-Kmin)*(1-b))*p*n
    # print([n,d])
    ret = (R*np.sqrt(n) - b*D*np.sqrt(d))*np.sqrt(d*n) - (Kmin+(K-Kmin)*(1-b))*p*n
    return ret

def aEq(state,param):

    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    ret=(Kmin+(K-Kmin)*(1-b))*p*n-b*E*a

    return ret

###############################################################################
# runge kutta solver

def solver_RK4(state,param,dt):

    k1p=dt*pEq(state,param);k1n=dt*nEq(state,param);k1a=dt*aEq(state,param);
    k1vec=np.array([k1p,k1n,k1a])
    state1=state+0.5*k1vec

    k2p=dt*pEq(state1,param);k2n=dt*nEq(state1,param);k2a=dt*aEq(state1,param);
    k2vec=np.array([k2p,k2n,k2a])
    state2=state+0.5*k2vec

    k3p=dt*pEq(state2,param);k3n=dt*nEq(state2,param);k3a=dt*aEq(state2,param);
    k3vec=np.array([k3p,k3n,k3a])
    state3=state+k3vec

    k4p=dt*pEq(state3,param);k4n=dt*nEq(state3,param);k4a=dt*aEq(state3,param);
    k4vec=np.array([k4p,k4n,k4a])

    var_state=np.array((k1vec+2*k2vec+2*k3vec+k4vec)/6)

    return state+var_state

###############################################################################
# functions for root finding

def initial_guess(param):

    T=100;dt=0.01;t=0
    state=np.array([0.01,0.98,0.01])
    state_new=np.zeros(3)

    while t<T:

        state_new=solver_RK4(state,param,dt)

        deriv=(state_new-state)/dt
        if np.sqrt(np.dot(deriv,deriv))<1e-6:# steadystate reached
            break

        # collapse=np.dot(state,state)
        # print([np.dot(deriv,deriv),state])
        if state[1]<1e-4:#collapse reached
            break
        if 1-state[1]-state[2]<1e-4:
            break

        state=state_new
        print(state)
        t=t+dt

    return state

def jacobian(state,param):

    p=state[0];n=state[1];a=state[2]
    dstate=[1e-6,1e-6,1e-6]
    dp=dstate[0];dn=dstate[1];da=dstate[2]

    # states with the added perturbation
    pstate=[p+dp,n,a]
    nstate=[p,n+dn,a]
    astate=[p,n,a+da]

    pert_state=[pstate,nstate,astate]

    Jac=np.zeros((3,3))
    for i in range(3):
        Jac[0,i]=(pEq(pert_state[i],param)-pEq(state,param))/dstate[i]
        Jac[1,i]=(nEq(pert_state[i],param)-nEq(state,param))/dstate[i]
        Jac[2,i]=(aEq(pert_state[i],param)-aEq(state,param))/dstate[i]

    ret = Jac
    return ret

def analytical_jacobian(state,param):

    p=state[0];n=state[1];a=state[2]
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    Jac=np.zeros((3,3))

    d=1-a-n
    C=Kmin+(K-Kmin)*(1-b)
    Y=foodProduction(state,param)
    dYN=b*b*(1-b)*Q*np.sqrt(a)
    dYA=b*b + Q*b*(1-b)*0.5*( n/np.sqrt(a) + (1-b)*(1-n)/np.sqrt(a) - 3*(1-b)*np.sqrt(a) )
    # dYA=b*b + Q*b*(1-b)*( 0.5*(b*n+(1-a)*(1-b))*np.sqrt(a) - np.sqrt(a)*(1-b)*(1-b) )

    Jac[0,0]=1-2*p/Y
    Jac[0,1]=(p/Y)*(p/Y)*dYN
    Jac[0,2]=(p/Y)*(p/Y)*dYA

    Jac[1,0]=-C*n
    Jac[1,1]=-C*p + R*np.sqrt(d)*(1-0.5*n/d) + D*b*np.sqrt(n)*(1-0.5*d/n)
    Jac[1,2]=D*b*np.sqrt(n)*(1-0.5*(R/D/b)*np.sqrt(n/d))

    Jac[2,0]=C*n
    Jac[2,1]=C*p
    Jac[2,2]=-E*b

    ret=Jac
    return ret

def savearray(data,array):

    array=np.concatenate((array,data),axis=0)
    return array

def fun(state,param):

    return [pEq(state,param), nEq(state,param), aEq(state,param)]

def findcriticalbeta(param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    bmin=0.1
    bmax=0.9

    b=bmin
    param=[b,K,R,D,E,Q,Kmin]

    old_fp=initial_guess(param)
    print(old_fp)
    old_stability=1

    db=0.005
    while b<bmax:

        param=[b,K,R,D,E,Q,Kmin]

        state0=old_fp

        # solve the root problem and get the fixed point
        sol = optimize.root(fun,state0,args=(param,),method='lm',jac=analytical_jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

        fixed_point = sol.x # steady states

        # test if the root finder actually converged to a fp
        btest=np.array(fun(fixed_point,param))
        l2test=np.sqrt(np.dot(btest,btest))
        if(l2test>1e-6):
            print("root finder didn't converge")
            print(b)
            break
        old_fp=np.array(fixed_point)

        # get the eigenvalues of the analytical_jacobian matrix to calculate stability
        jac=analytical_jacobian(fixed_point,param)
        la, v =linalg.eig(jac)
        stability=0
        if all(la.real<0):
            stability=1

        if stability!=old_stability:
            print("stability change")
            break

        b=b+db

    bc1=b
    bc2=b

    if bc1<bmax-2*db:

        b=bmax
        param=[b,K,R,D,E,Q,Kmin]

        old_fp=initial_guess(param)
        old_stability=1

        print(old_fp)
        db=0.005
        while b>bmin:
            print(b)

            param=[b,K,R,D,E,Q,Kmin]

            state0=old_fp

            # solve the root problem and get the fixed point
            sol = optimize.root(fun,state0,args=(param,),method='lm',jac=analytical_jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

            fixed_point = sol.x # steady states

            # test if the root finder actually converged to a fp
            btest=np.array(fun(fixed_point,param))
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test>1e-6):
                print("root finder didn't converge")
                break
            old_fp=np.array(fixed_point)

            # get the eigenvalues of the analytical_jacobian matrix to calculate stability
            jac=analytical_jacobian(fixed_point,param)
            la, v =linalg.eig(jac)
            stability=0
            if all(la.real<0):
                stability=1

            if stability!=old_stability:
                print("stability change")
                break

            b=b-db

        bc2=b

    deltab=bc2-bc1
    return deltab

def Plot(datafull):
    sns.set_context('paper')
    style.use('seaborn-paper')
    fig, (ax1) = plt.subplots(1,1)
    sns.set_style("ticks")

    data1=datafull[0]
    data2=datafull[1]
    data3=datafull[2]

    R1=data1[:,0]
    deltab1=data1[:,1]
    R2=data2[:,0]
    deltab2=data2[:,1]
    R3=data3[:,0]
    deltab3=data3[:,1]

    ax1.plot(R1,deltab1,'o',linewidth=2,alpha=0.7,label='D=0.5')
    ax1.plot(R2,deltab2,'o',linewidth=2,alpha=0.7,label='D=1.0')
    ax1.plot(R3,deltab3,'o',linewidth=2,alpha=0.7,label='D=1.5')

    fig.legend()
    sns.despine()
    ax1.set(ylabel=r"$\Delta \beta$")
    ax1.set(xlabel=r'$r$');
    plt.show()
    plt.close()

###############################################################################

R=1.0;D=1.0;E=1.0;b=0;K=4.5;Kmin=0.5;Q=1.0


Rmin=0.5; Rmax=1.5
Rarray=np.linspace(Rmin,Rmax,10)
Darray=np.array([0.5,1.0,1.5])

res1=np.array([[],[]]).T

D=1.5
print(D)
for R in Rarray:
    print(R)
    param=[b,K,R,D,E,Q,Kmin]
    deltab=findcriticalbeta(param)
    # print(R)
    # print(deltab)
    data=np.array([np.append(R,deltab)])
    res1=savearray(data,res1)

res2=np.array([[],[]]).T

D=0.75
print(D)
for R in Rarray:
    param=[b,K,R,D,E,Q,Kmin]
    deltab=findcriticalbeta(param)
    # print(R)
    # print(deltab)
    data=np.array([np.append(R,deltab)])
    res2=savearray(data,res2)

res3=np.array([[],[]]).T

D=1.0
print(D)
for R in Rarray:
    param=[b,K,R,D,E,Q,Kmin]
    deltab=findcriticalbeta(param)
    # print(R)
    # print(deltab)
    data=np.array([np.append(R,deltab)])
    res3=savearray(data,res3)

res=[res1,res2,res3]
Plot(res)
