# Creates and modifies attractor database for henon map
import attract as atr
import numpy as np
import matplotlib.pyplot as plt

# the Henon map
def henon(x, a, b):
    return np.array([1.0  - a*x[0]**2 + x[1], b*x[0]])

henon_db = atr.AttractorDB(db_path = '../../data/henon_attractor_14_3_4.h5', func = henon, dim = 2, a = 1.4, b = 0.3)
#henon_db.add_new_paths(num_paths = 1, length = 1000)
#henon_db.plot_path2D(9)

henon_db.add_new_pts(1000000)
henon_db.collect_seeds(num_seeds = 5000)
henon_db.tessellate()
henon_db.assign_pts_to_cells()
