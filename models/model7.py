# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot as plot
import attract as atr

"""
A 2D non-linear problem (Henon map)
"""
# set parameters
a, b = 1.4, 0.3
eps, delta = 0.0, 0.01
mu, id, zero =  np.zeros(2), np.identity(2), np.zeros(2)

# assign an attractor point as the starting point
def henon(x):
    return np.array([1.0  - a*x[0]**2 + x[1], b*x[0]])
data_dir = str(script_dir.parent.parent) + '/data/'
#x0 = (0.5086958043266177, 0.126731276358114) #small
#x0 = (0.11592029238259197, 0.20929831531734902) #medium
#x0 = (1.2546489681198767, 0.021540156609374392) #large
x0 = atr.AttractorDB(db_path = data_dir + 'henon_db_single.h5', func = henon, dim = 2).burn_in()

# create a deterministic Markov chain
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(x0, 0.01*id))
process_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, eps*id))
func_h = lambda k, x, noise: np.array([1.0 - a*x[0]**2 + x[1], b*x[0]]) + noise
conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero), cov = eps*id)

# define the observation model
H = np.array([[1.0, 0.0],[0.0, 1.0]])
func_o = lambda k, x, noise: np.dot(H, x) + noise
observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, delta*id))
conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(0, condition, mu), cov = delta*id)

# generates a trajectory according to the dynamic model
def gen_path(length):
    path = np.zeros((length, 2), dtype = 'float64')
    x = x0
    for i in range(length):
        path[i] = x
        x = func_h(i, x, zero)
    return path

# creates a Model object to feed the filter / combine the models
def model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = eps*id, noise_sim = process_noise, conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = delta*id, noise_sim = observation_noise, conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om), a, b, x0
