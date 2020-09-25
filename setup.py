import numpy as np

import units

def setup(N,Mpt,config):
    if config == 'cloud':
        pos,vel = set_cloud(N)
    elif config == 'disc':
        pos,vel = set_disc(N,Mpt)
    else:
        print('Unrecognised config option, aborting.')
        quit()
    return pos,vel

def set_disc(N,Mpt):
    """
    Setup particles in a rotating disc
        Parameters:
            N (int) : Number of particles
        return:
            pos (np.array, Nx3): Particle positions
            vel (np.array, Nx3): Particle velocities
    """
    type = np.ones(N)
    type[0] = 0

    gas = type[0] == 1
    ptmass = type[0] == 0

    pos = np.random.randn(N,3)   # randomly selected positions and velocities
    pos[:,2] = 0 # set z positions to 0

    dr = np.sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1] + pos[:,2]*pos[:,2])
    phi  = np.arctan2(pos[:,1],pos[:,0])

    vmod = np.sqrt(Mpt/dr)
    vel = np.stack([-vmod*np.sin(phi), vmod*np.cos(phi), np.zeros(N)]).T

    return pos, vel

def set_cloud(N):
    """
    Setup particles in a random sphere with no velocity
        Parameters:
            N (int) : Number of particles
        return:
            pos (np.array, Nx3): Particle positions
            vel (np.array, Nx3): Particle velocities
    """
    pos = np.random.randn(N,3)   # randomly selected positions and velocities
    vel = np.zeros(pos.shape)
    return pos, vel
