import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from multiprocessing import Pool
import time as TIME

from kernel import kernel
from dens import dens
from force import force
from setup import *
import units

class sphJC:
    @staticmethod
    def main():
        """ N-body simulation """
        # Simulation parameters
        t         = 0       # current time of the simulation
        tEnd      = 120     # time at which simulation ends
        dt        = 0.04    # timestep
        N         = 10000   # Number of particles
        Npt       = 1       # Number of point mass particles (star)
        M         = 1e-10   # mass in particles
        Mpt       = 2       # mass in star
        R         = 1.0     # size (radius) of system
        h         = 0.1     # smoothing length
        k         = 0.1     # equation of state constant
        n         = 1       # polytropic index
        nu        = 1e-10   # damping
        plotRealTime = True # switch on for plotting as the simulation goes along
        config    = 'disc'  # supported options are 'disc' or 'cloud'
        nproc     = 4       # number of multiprocessing threads

        # Generate Initial Conditions
        np.random.seed(52)            # set the random number generator seed

        lmbda = 2*k*(1+n)*np.pi**(-3/(2*n))*(M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n)/R**2  # ~ 2.01

        m = M/N                    # single particle mass

        pos,vel = setup(N,Mpt,config)

        # calculate initial gravitational accelerations
        pool = Pool(processes=nproc)
        if __name__=='__main__':
            pos_ = np.array_split(pos,nproc)
            vel_ = np.array_split(vel,nproc)
            acc = pool.map(force.getAcc, [(pos_[i],vel_[i],m,Mpt,h,k,n,lmbda,nu) for i in range(nproc)])
        acc = np.concatenate(acc)

        # number of timesteps
        Nt = int(np.ceil(tEnd/dt))

        # prep figure
        fig = plt.figure(figsize=(8,10), dpi=80)
        grid = plt.GridSpec(5, 1, wspace=0.0, hspace=0.5)
        ax1 = plt.subplot(grid[0:2,0])
        ax2 = plt.subplot(grid[2,0])
        ax3 = plt.subplot(grid[3:5,0])
        rr = np.zeros((100,3))
        rlin = np.linspace(0,1,100)
        rr[:,0] =rlin
        rho_analytic = lmbda/(4*k) * (R**2 - rlin**2)

        time = []
        L = []

        # Simulation Main Loop
        for i in range(Nt):
            tstart = TIME.time()
            # (1/2) kick
            vel += acc*dt/2
            # drift
            pos += vel*dt
            # update accelerations
            if __name__=='__main__':
                pos_ = np.array_split(pos,nproc)
                vel_ = np.array_split(vel,nproc)
                acc = pool.map(force.getAcc, [(pos_[i],vel_[i],m,Mpt,h,k,n,lmbda,nu) for i in range(nproc)])
            acc = np.concatenate(acc)
            # (1/2) kick
            vel += acc*dt/2
            # update time
            t += dt
            # get density for plotting
            if __name__=='__main__':
                pos_ = np.array_split(pos,nproc)
                rr_ = np.array_split(rr,nproc)
                rho = pool.map(dens.getDensity, [(pos_[i],pos_[i],m,h) for i in range(nproc)])
                rho_radial = pool.map(dens.getDensity,[(rr_[i],pos_[i],m,h) for i in range(nproc)])
            rho = np.concatenate(rho)
            rho_radial = np.concatenate(rho_radial)
            # plot in real time
            if plotRealTime or (i == Nt-1):
                # Plot particle position
                plt.sca(ax1)
                plt.cla()
                cval = np.minimum((rho-3)/3,1).flatten()
                plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
                ax1.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
                ax1.set_aspect('equal', 'box')
                ax1.set_xticks([-1,0,1])
                ax1.set_yticks([-1,0,1])
                ax1.set_facecolor('black')
                ax1.set_facecolor((.1,.1,.1))
                ax1.set_xlabel('x, AU')
                ax1.set_ylabel('y, AU')
                ax1.text(-1.5,3,
                         'tot simulation time: %.1f yrs' %(t*units.utime/365.25/24./60./60.),
                         fontsize=12)
                ax1.text(-1.5,2.5,'time for iteration: %.3f s' %(TIME.time()-tstart),
                         fontsize=12)

                # Density plot
                plt.sca(ax2)
                plt.cla()
                plt.xlabel('Radius, AU')
                plt.ylabel(r'Density, M$_{\odot}$AU$^{-3}$')
                ax2.set(xlim=(0, 1), ylim=(0, 3))
                ax2.set_aspect(0.1)
                plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
                plt.plot(rlin, rho_radial, color='blue')
                plt.pause(0.001)

                # Angular momentum plot
                AM = m*np.cross(vel,pos)
                L.append(np.linalg.norm(np.sum(AM,axis=0)))
                time.append(t)
                plt.sca(ax3)
                ax3.set_xlabel('time, yrs')
                ax3.set_ylabel(r'Angular momentum, $L=mv\times r$')
                plt.plot(np.asarray(time)*units.utime/365.25/24./60./60.,
                         L,color='green')
                ax3.set(ylim=(0, np.max(L)*3),
                        xlim=(0,tEnd*units.utime/365.25/24./60./60.))
                plt.pause(0.001)


        # add labels/legend
        plt.sca(ax2)

        plt.sca(ax3)

        plt.show()
        return

if __name__== "__main__":
    sph = sphJC()
    sph.main()
