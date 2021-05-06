import matplotlib.pyplot as plt
import numpy as np

def gasp_cohn(x, y, c):
    """
    Gaspari-Cohn taper function
    """
    r = abs(x - y) / c
    if r >= 0. and r < 1.:
        return 1. - 5./3. * r**2 + 5./8.* r**3 + r**4 / 2. - r**5/4.
    elif r >= 1. and r < 2.:
        return 4. - 5.*r + 5./3. * r**2 + 5./8.* r**3 - r**4 / 2. + r**5/12. - 2./(3. * r)
    else:
        return 0.


gc = lambda x: gasp_cohn(x, 0., 4.)

x = np.arange(-10., 10., 1)
y = np.array([gc(e) for e in x])
plt.plot(x, y)
plt.show()