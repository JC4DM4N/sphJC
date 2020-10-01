import numpy as np

from kernel import kernel

class dens:
    """
    Calculates the density from the smoothing length
    """
    @staticmethod
    def getSeparations(ri,rj):
        """
        Calcultes the pairwise separations between two particles
            Parameters:
                ri (np.array, 1x3): 3D vector position of particle i
                rj (np.array, 1x3): 3D vector position of particle j
            Retunrs:
                dx, dy, dz: the vector separation between the two particles
        """
        M = ri.shape[0]
        N = rj.shape[0]
        # positions ri = (x,y,z)
        rix = ri[:,0].reshape((M,1))
        riy = ri[:,1].reshape((M,1))
        riz = ri[:,2].reshape((M,1))
        # other set of points positions rj = (x,y,z)
        rjx = rj[:,0].reshape((N,1))
        rjy = rj[:,1].reshape((N,1))
        rjz = rj[:,2].reshape((N,1))
        # matrices that store all pairwise particle separations: r_i - r_j
        dx = rix - rjx.T
        dy = riy - rjy.T
        dz = riz - rjz.T

        return dx, dy, dz

    @staticmethod
    def getDensity(inp):
        """
        Calculates the density at sampling loctions from SPH particle distribution
            Parameters:
                r (np.array, Mx3): matrix of sampling locations
                pos (np.array, Nx3): matrix of SPH particle positions
                m (float) : particle mass
                h (float) : smoothing length
            Returns:
                rho (np.array, Mx1) : vector of accelerations
        """
        r,pos,m,h = inp

        M = r.shape[0]
        dx, dy, dz = dens.getSeparations(r,pos)
        rho = np.sum(m*kernel.W(dx,dy,dz,h),1).reshape((M,1))

        return rho

    @staticmethod
    def getPressure(rho,k,n):
        """
        Calculates the equation of State
            Parameters:
                rho (np.array, 1x3): particle densities
                k (float) : equation of state constant
                n (float) : polytropic index
            Returns:
                P (float) : pressure
        """
        P = k*rho**(1+1/n)
        return P
