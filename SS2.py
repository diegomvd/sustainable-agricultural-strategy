# -*- coding: utf-8 -*-

"""
TEMPORAL DYANMICS PLOTS IN THE K,I PLANE
"""

import numpy as np
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
    if a<0:
        a=0
    if d<0:
        d=0
    if n<0:
        n=0

    effective_land=(n+d*(1-b))
    prod=b*(b*a+Q*(1-b)*effective_land*np.power(a,0.5))

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

    if a<0:
        a=0
    if d<0:
        d=0
    if n<0:
        n=0
    ret = (R*np.power(n,0.5)-b*D*np.power(d,0.5))*np.power(d*n,0.5)-(Kmin+(K-Kmin)*(1-b))*p*n
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

    T=100;dt=0.001;t=0
    state=np.array([0.001,0.98,0.01])
    state_new=np.zeros(3)

    while t<T:

        state_new=solver_RK4(state,param,dt)
        state=state_new

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

def savearray(data,array):

    array=np.concatenate((array,data),axis=0)
    return array

def fun(state,param):

    return [pEq(state,param), nEq(state,param), aEq(state,param)]

def findSSbeta(param,db_save):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    bmin=0.05
    bmax=0.95

    dataarr=np.array([[],[],[],[],[]]).T

    b=bmin
    param=[b,K,R,D,E,Q,Kmin]
    old_fp=initial_guess(param)
    old_stability=1

    b_save=b
    db=0.001
    print('main branch')
    # here we recover the initially stable branch and follow it
    # two things can happen:
    # 1- there's a fold bifurcation => root finder wont converge and i need to
    # stop go to bmax and repeat
    # 2 - i can recover the whole branch, everything is ok
    # once i have all of the main branches i go from scratch and recover the
    # unstable. the two same things can happen
    fold=0
    while b<bmax:

        param=[b,K,R,D,E,Q,Kmin]

        state0=old_fp

        # solve the root problem and get the fixed point
        sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

        fixed_point = sol.x # steady states

        # test if the root finder actually converged to a fp
        btest=np.array(fun(fixed_point,param))
        l2test=np.sqrt(np.dot(btest,btest))
        if(l2test>1e-6):
            print("root finder didn't converge")
            fold=1
            break
        old_fp=np.array(fixed_point)

        # get the eigenvalues of the jacobian matrix to calculate stability
        jac=jacobian(fixed_point,param)
        la, v =linalg.eig(jac)
        stability=0
        if all(la.real<0):
            stability=1

        old_stability=stability

        if b>=b_save:
            b_save+=db_save
            fixed_point_save=np.append(fixed_point,stability)
            data=np.array([np.append(b,fixed_point_save)])
            dataarr=savearray(data,dataarr)

        b=b+db

    # print(dataarr[:,0])

    if fold==1: # if there was a fold bif, i go to b=bmax and run backwards
        dataarrfold=np.array([[],[],[],[],[]]).T
        b=bmax
        param=[b,K,R,D,E,Q,Kmin]
        old_fp=initial_guess(param)
        old_stability=1

        b_save=b
        db=0.001
        print('main branch after fold')

        while b>bmin:

            # print(b)
            param=[b,K,R,D,E,Q,Kmin]

            state0=old_fp

            # solve the root problem and get the fixed point
            sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

            fixed_point = sol.x # steady states

            # test if the root finder actually converged to a fp
            btest=np.array(fun(fixed_point,param))
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test>1e-6):
                print("root finder didn't converge")
                break
            old_fp=np.array(fixed_point)

            # get the eigenvalues of the jacobian matrix to calculate stability
            jac=jacobian(fixed_point,param)
            la, v =linalg.eig(jac)
            stability=0
            if all(la.real<0):
                stability=1

            old_stability=stability

            if b<=b_save:
                b_save-=db_save
                fixed_point_save=np.append(fixed_point,stability)
                data=np.array([np.append(b,fixed_point_save)])
                dataarrfold=savearray(data,dataarrfold)

            b=b-db

        dataarrfoldflip=np.flip(dataarrfold,axis=0)
        dataarr=savearray(dataarrfoldflip,dataarr)

    # print(dataarr[:,0])
    # we now do the same for the unstable branch

    dataarru=np.array([[],[],[],[],[]]).T

    b=bmin
    param=[b,K,R,D,E,Q,Kmin]

    old_fp=np.array([0.01,0.1,0.1])#initial_guess(param)-np.array([0.025,0.2,0.2])
    old_stability=1

    b_save=b
    db=0.001
    print('unstable branch')
    fold=0
    while b<bmax:

        param=[b,K,R,D,E,Q,Kmin]

        state0=old_fp

        # solve the root problem and get the fixed point
        sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

        fixed_point = sol.x # steady states

        # test if the root finder actually converged to a fp
        btest=np.array(fun(fixed_point,param))
        l2test=np.sqrt(np.dot(btest,btest))
        if(l2test>1e-6):
            print("root finder didn't converge")
            fold=1
            break
        old_fp=np.array(fixed_point)

        # get the eigenvalues of the jacobian matrix to calculate stability
        jac=jacobian(fixed_point,param)
        la, v =linalg.eig(jac)
        stability=0
        if all(la.real<0):
            stability=1

        if old_stability!=stability:
            print("cool we found it")
        old_stability=stability

        if b>=b_save:
            b_save+=db_save
            fixed_point_save=np.append(fixed_point,stability)
            data=np.array([np.append(b,fixed_point_save)])
            dataarru=savearray(data,dataarru)

        b=b+db

    if fold==1: # if there was a fold bif, i go to b=bmax and run backwards
        dataarrufold=np.array([[],[],[],[],[]]).T
        b=bmax
        param=[b,K,R,D,E,Q,Kmin]
        old_fp=initial_guess(param)-np.array([0.0,0.2,0.0])
        old_stability=1

        b_save=b
        db=0.001
        print('unstable branch after fold')

        while b>bmin:

            # print(b)
            param=[b,K,R,D,E,Q,Kmin]

            state0=old_fp

            # solve the root problem and get the fixed point
            sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

            fixed_point = sol.x # steady states

            # test if the root finder actually converged to a fp
            btest=np.array(fun(fixed_point,param))
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test>1e-6):
                print("root finder didn't converge")
                break
            old_fp=np.array(fixed_point)

            # get the eigenvalues of the jacobian matrix to calculate stability
            jac=jacobian(fixed_point,param)
            la, v =linalg.eig(jac)
            stability=0
            if all(la.real<0):
                stability=1

            old_stability=stability

            if b<=b_save:
                b_save-=db_save
                fixed_point_save=np.append(fixed_point,stability)
                data=np.array([np.append(b,fixed_point_save)])
                dataarrufold=savearray(data,dataarrufold)

            b=b-db

        dataarrufoldflip=np.flip(dataarrufold,axis=0)
        dataarru=savearray(dataarrufoldflip,dataarru)

    print(dataarru[:,0])
    ret=[dataarr,dataarru]

    return ret

def PlotBifurcation(data1,data2,data3,data4):
    sns.set_context('paper')
    style.use('seaborn-paper')
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,sharex='row',sharey='col',gridspec_kw={'hspace':0.04,'wspace': 0.2})
    sns.set_style("ticks")

    b=data1[0][:,0]
    p=data1[0][:,1]
    n=data1[0][:,2]
    a=data1[0][:,3]
    d=np.ones(len(a))-n-a
    s=data1[0][:,4]

    # print(s)

    bu=data1[1][:,0]
    pu=data1[1][:,1]
    nu=data1[1][:,2]
    au=data1[1][:,3]
    du=np.ones(len(au))-nu-au
    su=data1[1][:,4]

    # print(bu)
    print(bu)
    foldind=np.where((bu[1:]-bu[:-1])>0.002)
    ii=foldind[0][0]
    print(ii)
    ii=ii+1

    # print(bu[:ii])
    # print(bu[ii:])
    # unstable branches
    ax1.plot(bu[:ii],pu[:ii],linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
    ax2.plot(bu[:ii],nu[:ii],linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
    ax2.plot(bu[:ii],au[:ii],linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
    # ax2.plot(bu[:ii],du[:ii],linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);
    ax1.plot(bu[ii:],pu[ii:],linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
    ax2.plot(bu[ii:],nu[ii:],linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
    ax2.plot(bu[ii:],au[ii:],linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
    # ax2.plot(bu[ii:],du[ii:],linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);

    unstab_ind=np.where(s<1);
    # print(unstab_ind[0])
    if unstab_ind[0].size>0:
        c1=unstab_ind[0][0];c2=unstab_ind[0][-1]+1

        b1=b[:c1];b2=b[c1:c2];b3=b[c2:]
        p1=p[:c1];p2=p[c1:c2];p3=p[c2:]
        n1=n[:c1];n2=n[c1:c2];n3=n[c2:]
        a1=a[:c1];a2=a[c1:c2];a3=a[c2:]
        d1=d[:c1];d2=d[c1:c2];d3=d[c2:]

        foldind=np.where((b2[1:]-b2[:-1])>0.002)
        print(foldind)
        ii=foldind[0][0]
        ii=ii+1
        #population plot
        ax1.plot(b1,p1,linewidth=2,color='tab:blue',alpha=0.7);
        ax1.plot(b2[:ii],p2[:ii],linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
        ax1.plot(b2[ii:],p2[ii:],linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
        ax1.plot(b3,p3,linewidth=2,color='tab:blue',alpha=0.7)
        ax1.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:blue',alpha=0.7)
        ax1.vlines(b[c1],0,p[c1],color='0.5',linestyles='dashed');
        ax1.vlines(b[c2],0,p[c2],color='0.5',linestyles='dashed');
        ax1.arrow(b[c1],0.05,b[c2]-b[c1],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax1.arrow(b[c2],0.05,b[c1]-b[c2],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax1.annotate(s=r'$\Delta \beta$', xy=(0.57,0.06))

        #land plot
        ax2.plot(b1,n1,linewidth=2,color='tab:green',label=r"$N^{\star}$",alpha=0.7);
        ax2.plot(b2[:ii],n2[:ii],linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
        ax2.plot(b2[ii:],n2[ii:],linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
        ax2.plot(b3,n3,linewidth=2,color='tab:green',alpha=0.7)

        ax2.plot(b1,a1,linewidth=2,color='tab:orange',label=r"$A^{\star}$",alpha=0.7);
        ax2.plot(b2[:ii],a2[:ii],linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
        ax2.plot(b2[ii:],a2[ii:],linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
        ax2.plot(b3,a3,linewidth=2,color='tab:orange',alpha=0.7)

        # ax2.plot(b1,d1,linewidth=2,color='tab:red',label=r"$D^{\star}$",alpha=0.7);
        # ax2.plot(b2[:ii],d2[:ii],linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);
        # ax2.plot(b2[ii:],d2[ii:],linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);
        # ax2.plot(b3,d3,linewidth=2,color='tab:red',alpha=0.7)

        ax2.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:green',alpha=0.7)
        # ax2.plot(b2,np.ones(len(b2)),linewidth=2,color='tab:red',alpha=0.7)
        ax2.vlines(b[c1],0,n[c1],color='0.5',linestyles='dashed');
        ax2.vlines(b[c2],0,n[c2],color='0.5',linestyles='dashed');

    else:

        ax1.plot(b,p,linewidth=2,color='tab:blue',label=r"$P^{\star}$",alpha=0.7);
        ax2.plot(b,n,linewidth=2,color='tab:green',label=r"$N^{\star}$",alpha=0.7);
        ax2.plot(b,a,linewidth=2,color='tab:orange',label=r"$A^{\star}$",alpha=0.7);
        # ax2.plot(b,d,linewidth=2,color='tab:red',label=r"$D^{\star}$",alpha=0.7);

##############################################################################
# data2

    b=data2[0][:,0]
    p=data2[0][:,1]
    n=data2[0][:,2]
    a=data2[0][:,3]
    d=np.ones(len(a))-n-a
    s=data2[0][:,4]

    bu=data2[1][:,0]
    pu=data2[1][:,1]
    nu=data2[1][:,2]
    au=data2[1][:,3]
    du=np.ones(len(au))-nu-au
    su=data2[1][:,4]

    # unstable branches
    ax3.plot(bu,pu,linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
    ax4.plot(bu,nu,linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
    ax4.plot(bu,au,linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
    # ax4.plot(bu,du,linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);

    unstab_ind=np.where(s<1);
    # print(unstab_ind[0])
    if unstab_ind[0].size>0:
        c1=unstab_ind[0][0];c2=unstab_ind[0][-1]+1

        b1=b[:c1];b2=b[c1:c2];b3=b[c2:]
        p1=p[:c1];p2=p[c1:c2];p3=p[c2:]
        n1=n[:c1];n2=n[c1:c2];n3=n[c2:]
        a1=a[:c1];a2=a[c1:c2];a3=a[c2:]
        d1=d[:c1];d2=d[c1:c2];d3=d[c2:]

        #population plot
        ax3.plot(b1,p1,linewidth=2,color='tab:blue',alpha=0.7);
        ax3.plot(b2,p2,linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
        ax3.plot(b3,p3,linewidth=2,color='tab:blue',alpha=0.7)
        ax3.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:blue',alpha=0.7)
        ax3.vlines(b[c1],0,p[c1],color='0.5',linestyles='dashed');
        ax3.vlines(b[c2],0,p[c2],color='0.5',linestyles='dashed');
        ax3.arrow(b[c1],0.05,b[c2]-b[c1],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax3.arrow(b[c2],0.05,b[c1]-b[c2],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax3.annotate(s=r'$\Delta \beta$', xy=(0.57,0.06))

        #land plot
        ax4.plot(b1,n1,linewidth=2,color='tab:green',alpha=0.7);
        ax4.plot(b2,n2,linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
        ax4.plot(b3,n3,linewidth=2,color='tab:green',alpha=0.7)

        ax4.plot(b1,a1,linewidth=2,color='tab:orange',alpha=0.7);
        ax4.plot(b2,a2,linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
        ax4.plot(b3,a3,linewidth=2,color='tab:orange',alpha=0.7)

        # ax4.plot(b1,d1,linewidth=2,color='tab:red',alpha=0.7);
        # ax4.plot(b2,d2,linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);
        # ax4.plot(b3,d3,linewidth=2,color='tab:red',alpha=0.7)

        ax4.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:green',alpha=0.7)
        # ax4.plot(b2,np.ones(len(b2)),linewidth=2,color='tab:red',alpha=0.7)
        ax4.vlines(b[c1],0,n[c1],color='0.5',linestyles='dashed');
        ax4.vlines(b[c2],0,n[c2],color='0.5',linestyles='dashed');

    else:
        ax3.plot(b,p,linewidth=2,color='tab:blue',alpha=0.7);
        ax4.plot(b,n,linewidth=2,color='tab:green',alpha=0.7);
        ax4.plot(b,a,linewidth=2,color='tab:orange',alpha=0.7);
        # ax4.plot(b,d,linewidth=2,color='tab:red',alpha=0.7);

##############################################################################
# data3

    b=data3[0][:,0]
    p=data3[0][:,1]
    n=data3[0][:,2]
    a=data3[0][:,3]
    d=np.ones(len(a))-n-a
    s=data3[0][:,4]

    bu=data3[1][:,0]
    pu=data3[1][:,1]
    nu=data3[1][:,2]
    au=data3[1][:,3]
    du=np.ones(len(au))-nu-au
    su=data3[1][:,4]

    # unstable branches
    ax5.plot(bu,pu,linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
    ax6.plot(bu,nu,linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
    ax6.plot(bu,au,linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
    # ax6.plot(bu,du,linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);

    unstab_ind=np.where(s<1);
    # print(unstab_ind[0])
    if unstab_ind[0].size>0:
        c1=unstab_ind[0][0];c2=unstab_ind[0][-1]+1

        b1=b[:c1];b2=b[c1:c2];b3=b[c2:]
        p1=p[:c1];p2=p[c1:c2];p3=p[c2:]
        n1=n[:c1];n2=n[c1:c2];n3=n[c2:]
        a1=a[:c1];a2=a[c1:c2];a3=a[c2:]
        d1=d[:c1];d2=d[c1:c2];d3=d[c2:]

        #population plot
        ax5.plot(b1,p1,linewidth=2,color='tab:blue',alpha=0.7);
        ax5.plot(b2,p2,linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
        ax5.plot(b3,p3,linewidth=2,color='tab:blue',alpha=0.7)
        ax5.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:blue',alpha=0.7)
        ax5.vlines(b[c1],0,p[c1],color='0.5',linestyles='dashed');
        ax5.vlines(b[c2],0,p[c2],color='0.5',linestyles='dashed');
        ax5.arrow(b[c1],0.05,b[c2]-b[c1],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax5.arrow(b[c2],0.05,b[c1]-b[c2],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax5.annotate(s=r'$\Delta \beta$', xy=(0.57,0.06))

        #land plot
        ax6.plot(b1,n1,linewidth=2,color='tab:green',alpha=0.7);
        ax6.plot(b2,n2,linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
        ax6.plot(b3,n3,linewidth=2,color='tab:green',alpha=0.7)

        ax6.plot(b1,a1,linewidth=2,color='tab:orange',alpha=0.7);
        ax6.plot(b2,a2,linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
        ax6.plot(b3,a3,linewidth=2,color='tab:orange',alpha=0.7)

        # ax6.plot(b1,d1,linewidth=2,color='tab:red',alpha=0.7);
        # ax6.plot(b2,d2,linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);
        # ax6.plot(b3,d3,linewidth=2,color='tab:red',alpha=0.7)

        ax6.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:green',alpha=0.7)
        # ax6.plot(b2,np.ones(len(b2)),linewidth=2,color='tab:red',alpha=0.7)
        ax6.vlines(b[c1],0,n[c1],color='0.5',linestyles='dashed');
        ax6.vlines(b[c2],0,n[c2],color='0.5',linestyles='dashed');

    else:
        ax5.plot(b,p,linewidth=2,color='tab:blue',alpha=0.7);
        ax6.plot(b,n,linewidth=2,color='tab:green',alpha=0.7);
        ax6.plot(b,a,linewidth=2,color='tab:orange',alpha=0.7);
        # ax6.plot(b,d,linewidth=2,color='tab:red',alpha=0.7);

##############################################################################
# data4

    b=data4[0][:,0]
    p=data4[0][:,1]
    n=data4[0][:,2]
    a=data4[0][:,3]
    d=np.ones(len(a))-n-a
    s=data4[0][:,4]

    bu=data4[1][:,0]
    pu=data4[1][:,1]
    nu=data4[1][:,2]
    au=data4[1][:,3]
    du=np.ones(len(au))-nu-au
    su=data4[1][:,4]

    # unstable branches
    ax7.plot(bu,pu,linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
    ax8.plot(bu,nu,linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
    ax8.plot(bu,au,linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
    # ax8.plot(bu,du,linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);

    unstab_ind=np.where(s<1);
    # print(unstab_ind[0])
    if unstab_ind[0].size>0:
        c1=unstab_ind[0][0];c2=unstab_ind[0][-1]+1

        b1=b[:c1];b2=b[c1:c2];b3=b[c2:]
        p1=p[:c1];p2=p[c1:c2];p3=p[c2:]
        n1=n[:c1];n2=n[c1:c2];n3=n[c2:]
        a1=a[:c1];a2=a[c1:c2];a3=a[c2:]
        d1=d[:c1];d2=d[c1:c2];d3=d[c2:]

        #population plot
        ax7.plot(b1,p1,linewidth=2,color='tab:blue',alpha=0.7);
        ax7.plot(b2,p2,linewidth=2,color='tab:blue',linestyle='dashed',alpha=0.7);
        ax7.plot(b3,p3,linewidth=2,color='tab:blue',alpha=0.7)
        ax7.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:blue',alpha=0.7)
        ax7.vlines(b[c1],0,p[c1],color='0.5',linestyles='dashed');
        ax7.vlines(b[c2],0,p[c2],color='0.5',linestyles='dashed');
        ax7.arrow(b[c1],0.05,b[c2]-b[c1],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax7.arrow(b[c2],0.05,b[c1]-b[c2],0.0,color='0.2',linewidth=0.3,head_width=0.008, head_length=0.01, length_includes_head=True)
        ax7.annotate(s=r'$\Delta \beta$', xy=(0.57,0.06))

        #land plot
        ax8.plot(b1,n1,linewidth=2,color='tab:green',alpha=0.7);
        ax8.plot(b2,n2,linewidth=2,color='tab:green',linestyle='dashed',alpha=0.7);
        ax8.plot(b3,n3,linewidth=2,color='tab:green',alpha=0.7)

        ax8.plot(b1,a1,linewidth=2,color='tab:orange',alpha=0.7);
        ax8.plot(b2,a2,linewidth=2,color='tab:orange',linestyle='dashed',alpha=0.7);
        ax8.plot(b3,a3,linewidth=2,color='tab:orange',alpha=0.7)

        # ax8.plot(b1,d1,linewidth=2,color='tab:red',alpha=0.7);
        # ax8.plot(b2,d2,linewidth=2,color='tab:red',linestyle='dashed',alpha=0.7);
        # ax8.plot(b3,d3,linewidth=2,color='tab:red',alpha=0.7)

        ax8.plot(b2,np.zeros(len(b2)),linewidth=2,color='tab:green',alpha=0.7)
        # ax8.plot(b2,np.ones(len(b2)),linewidth=2,color='tab:red',alpha=0.7)
        ax8.vlines(b[c1],0,n[c1],color='0.5',linestyles='dashed');
        ax8.vlines(b[c2],0,n[c2],color='0.5',linestyles='dashed');

    else:
        ax7.plot(b,p,linewidth=2,color='tab:blue',alpha=0.7);
        ax8.plot(b,n,linewidth=2,color='tab:green',alpha=0.7);
        ax8.plot(b,a,linewidth=2,color='tab:orange',alpha=0.7);
        # ax8.plot(b,d,linewidth=2,color='tab:red',alpha=0.7);

    sns.despine()
    #fig.legend(loc='upper right', bbox_to_anchor=(0.4, 0.55))
    fig.legend()
    # text = ax.text(0.0,0.5,"Population", size=12, verticalalignment='center', rotation=270)
    ax1.set_title(r"Population")
    ax2.set_title(r"Land")
    # ax1.set(ylabel=r"r$=0.9$")
    # ax3.set(ylabel=r"r$=0.9714$")
    # ax5.set(ylabel=r"r$=1$")
    # ax7.set(ylabel=r"r$=2$")
    ax7.set(xlabel=r"$\beta$")
    ax8.set(xlabel=r"$\beta$")
    # ax1.set(ylabel=r"Population")
    # ax2.set(ylabel=r"Land")
    # ax2.set(xlabel=r'$\beta$');
    # ax4.set(xlabel=r'$\beta$');
    # ax6.set(xlabel=r'$\beta$');
    # ax8.set(xlabel=r'$\beta$');
    # ax1.set_title(r"$r=0.95$");
    # ax3.set_title(r"$r=0.9714$");
    # ax5.set_title(r"$r=1.0$");
    # ax7.set_title(r"$r=1.25$");
    #plt.tight_layout()
    plt.show()
    plt.close()

###############################################################################
#0.97139
D=1.0;E=1.0;b=0;K=4.5;Kmin=0.5;Q=1.0

R=0.9
param=[b,K,R,D,E,Q,Kmin]
db_save=0.001
equilibria1=findSSbeta(param,db_save)

R=0.97139
param=[b,K,R,D,E,Q,Kmin]
db_save=0.001
equilibria2=findSSbeta(param,db_save)

R=1.0
param=[b,K,R,D,E,Q,Kmin]
db_save=0.001
equilibria3=findSSbeta(param,db_save)

R=2.0
param=[b,K,R,D,E,Q,Kmin]
db_save=0.001
equilibria4=findSSbeta(param,db_save)

PlotBifurcation(equilibria1,equilibria2,equilibria3,equilibria4)



###############################################################################
# test=1
# while test>0:
#     # random perturbation
#     equilibrium=np.array([p1,n1,a1]) # this excludes population
#     varp=p1/10;varn=n1/10;vara=a1/10
#     cov=[[varp,0,0],[0,varn,0],[0,0,vara]]
#
#     old_fp=np.random.multivariate_normal(equilibrium,cov)
#
#     b_save=b
#     db=0.005
#     while b>bmin-db:
#
#         param=[b,K,R,D,E,Q,Kmin]
#
#         state0=old_fp
#
#         # solve the root problem and get the fixed point
#         sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})
#
#         fixed_point = sol.x # steady states
#
#         # test if the root finder actually converged to a fp
#         btest=np.array(fun(fixed_point,param))
#         l2test=np.sqrt(np.dot(btest,btest))
#         print(l2test)
#         if(l2test>1e-6):
#             print("root finder didn't converge")
#             break
#         old_fp=np.array(fixed_point)
#
#         # get the eigenvalues of the jacobian matrix to calculate stability
#         jac=jacobian(fixed_point,param)
#         la, v =linalg.eig(jac)
#         stability=0
#         if all(la.real<0):
#             stability=1
#         print(stability)
#
#         if stability!=old_stability:
#             print("found first unstable branch")
#
#         old_stability=stability
#
#         if b<=b_save:
#             b_save+=db_save
#             fixed_point_save=np.append(fixed_point,stability)
#             data=np.array([np.append(b,fixed_point_save)])
#             dataarrtest=savearray(data,dataarrtest)
#
#         b=b-db
#
#     print(dataarr[:c1,1:-1])
#     print(dataarrtest[:,:-1])
#     diff=dataarr[:c1,:-1]-dataarrtest[:,:-1]
#     if np.sqrt(np.dot(diff,diff))>1e-6:
#         test=0
#         print("unstable branch found")
#     else:
#         print("didn't find unstable branch")
#
