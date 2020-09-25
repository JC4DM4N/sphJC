import numpy as np

umass = 1.99e33
udist = 1.5e13
yr = 3.15576e7
#G = 6.672041e-8
G = 1
utime = np.sqrt((udist**3)/(G*umass))
