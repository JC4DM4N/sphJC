import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

from part import part
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
        N         = 400     # Number of particles
        Npt       = 1       # Number of point mass particles (star)
        M         = 2    # mass in particles
        Mpt       = 1       # mass in star
        R         = 1.0     # size (radius) of system
        h         = 0.1     # smoothing length
        k         = 0.1     # equation of state constant
        n         = 1       # polytropic index
        nu        = 1.0     # damping
        plotRealTime = True # switch on for plotting as the simulation goes along

        # Generate Initial Conditions
        np.random.seed(42)            # set the random number generator seed

        lmbda = 2*k*(1+n)*np.pi**(-3/(2*n))*(M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n)/R**2  # ~ 2.01
        m = M/N                    # single particle mass

        pos,vel = set_cloud(N)

        # calculate initial gravitational accelerations
        acc = force.getAcc(pos,vel,m,h,k,n,lmbda,nu)

        # number of timesteps
        Nt = int(np.ceil(tEnd/dt))

        # prep figure
        fig = plt.figure(figsize=(4,5), dpi=80)
        grid = plt.GridSpec(5, 1, wspace=0.0, hspace=0.3)
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
            # (1/2) kick
            vel += acc*dt/2
            # drift
            pos += vel*dt
            # update accelerations
            acc = force.getAcc(pos,vel,m,h,k,n,lmbda,nu)
            # (1/2) kick
            vel += acc*dt/2
            # update time
            t += dt
            # get density for plottiny
            rho = dens.getDensity(pos,pos,m,h)
            # plot in real time
            if plotRealTime or (i == Nt-1):
                plt.sca(ax1)
                plt.cla()
                cval = np.minimum((rho-3)/3,1).flatten()
                plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
                ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
                ax1.set_aspect('equal', 'box')
                ax1.set_xticks([-1,0,1])
                ax1.set_yticks([-1,0,1])
                ax1.set_facecolor('black')
                ax1.set_facecolor((.1,.1,.1))

                plt.sca(ax2)
                plt.cla()
                ax2.set(xlim=(0, 1), ylim=(0, 3))
                ax2.set_aspect(0.1)
                plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
                rho_radial = dens.getDensity(rr,pos,m,h)
                plt.plot(rlin, rho_radial, color='blue')
                plt.pause(0.001)

                r = np.sqrt(pos[:,0]*pos[:,0]+pos[:,1]*pos[:,1]+pos[:,2]*pos[:,2])
                v = np.sqrt(vel[:,0]*vel[:,0]+vel[:,1]*vel[:,1]+vel[:,2]*vel[:,2])

                AM = np.linalg.norm(m*np.cross(vel,pos),axis=1)
                L.append(np.sum(AM))
                time.append(t)
                plt.sca(ax3)
                plt.plot(time,L,color='green')
                plt.pause(0.001)


        # add labels/legend
        plt.sca(ax2)
        plt.xlabel('radius')
        plt.ylabel('density')
        # Save figure
        plt.savefig('plot.png',dpi=240)

        '''
        plt.figure()
        plt.plot(L,time)
        '''
        plt.show()
        return

if __name__== "__main__":
    sph = sphJC()
    sph.main()
