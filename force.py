import numpy as np

from dens import dens
from kernel import kernel

class force:
    """
    Calculates particle accelerations
    """
    @staticmethod
    def getAcc(pos,vel,m,h,k,n,lmbda,nu):
        """
        Calculates the acceleration on each SPH particle
        Parameters:
            pos (np.array, Nx3): matrix of N particle positions
            vel (np.array, Nx3): matrix of N particle velocities
            m (float): particle mass
            h (float): smoothing length
            k (float): equation of state constant
            n (float): polytropic index
            lmbda (float): external force constant
            nu (float): viscosity
        Returns:
            a (np.array, Nx3): particle accelerations
        """
        N = pos.shape[0]

        # Calculate densities at the position of the particles
        rho = dens.getDensity(pos,pos,m,h)

        # Get the pressures
        P = dens.getPressure(rho,k,n)

        # Get pairwise distances and gradients
        dx, dy, dz = dens.getSeparations(pos,pos)
        dWx, dWy, dWz = kernel.gradW(dx,dy,dz,h)

        # Add Pressure contribution to accelerations
        ax = -np.sum(m*(P/rho**2 + P.T/rho.T**2)*dWx, 1).reshape((N,1))
        ay = -np.sum(m*(P/rho**2 + P.T/rho.T**2)*dWy, 1).reshape((N,1))
        az = -np.sum(m*(P/rho**2 + P.T/rho.T**2)*dWz, 1).reshape((N,1))

        # pack together the acceleration components
        a = np.hstack((ax,ay,az))

        # Add external potential force and viscosity
        a += -lmbda*pos - nu*vel

        return a
