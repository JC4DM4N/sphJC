# sphJC
Smoothed particle hydrodynamics code in Python.

A fluid is represented as N pseudo-particles, each with a mass, position and velocity, and interact through gravitational, pressure and viscous forces.

Particle motion is calculated by solving the Euler equation for an ideal fluid. Every particle interatcs with every other particle through a Gaussian smoothing 
kernel, with a characterisitc smoothing length, h.

To run:

python sphJC.py
