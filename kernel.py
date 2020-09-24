import numpy as np

class kernel:
    """
    Calcualtes the SPH kernel properties
    """
    @staticmethod
    def W(x,y,z,h):
        """
        Calculates the 3D Gaussian smoothing kernel
            Parameteres:
                x : particle x position
                y : particle y position
                z : particle z position
                h : smoothing length
            Returns:
                w :is the evaluated smoothing function
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
        return w

    @staticmethod
    def gradW(x,y,z,h):
        """
        Calcualtes the gradient of the 3D Gausssian smoothing kernel
            Parameteres:
                x : particle x position
                y : particle y position
                z : particle z position
                h : smoothing length
            Returns:
                wx, wy, wz: the 3D gradient of the smoothing kernel
        """
        r = np.sqrt(x**2+y**2+z**2)
        n = -2*np.exp(-r**2/h**2)/h**5/(np.pi)**(3/2)
        wx = n*x
        wy = n*y
        wz = n*z
        return wx, wy, wz
