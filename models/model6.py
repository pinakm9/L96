# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_path = Path(dirname(realpath(__file__)))
module_dir = str(script_path.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot as plot

"""
A 2D non-linear problem (Henon map)
"""
# Create a Markov chain
a, b, eps  = 0.5, 0.1, 0.0
mu, id, zero =  np.zeros(2), np.identity(2), np.zeros(2)
prior = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal((0.5086958043266177, 0.126731276358114), 0.001*id))
process_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, eps*id))
func_h = lambda k, x, noise: np.array([1.0 - a*x[0]**2 + x[1], b*x[0]]) + noise
conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero), cov = eps*id)

# Define the observation model
delta = 0.01
H = np.array([[1.0, 1.0],[0.0, 2.0]])
func_o = lambda k, x, noise: np.dot(H, x) + 0.5*np.sin(x) + noise
observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, delta*id))
conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(0, condition, mu), cov = delta*id)

"""
# Define F and compute its minimum and gradient
def F(k, x, x_prev, observation):
    a = x - func_h(k, x_prev, zero)
    b = observation - func_o(k, x, zero)
    return 0.5*(np.dot(a.T, a)/eps + np.dot(b.T, b)/delta)

def grad_F(k, x, x_prev, observation):
    a = x - func_h(k, x_prev, zero)
    b = observation - func_o(k, x, zero)
    return a/eps - np.dot(b.T, H + 0.5*np.diag(np.cos(x)))/delta

A = np.identity(2)/eps + np.dot(H.T, H)/delta

def argmin_F(k, x_prev, observation):
    f = lambda x: F(k, x, x_prev, observation)
    x0 = np.linalg.solve(A, func_h(k, x_prev, zero)/eps + np.dot(H.T, observation)/delta)
    return scipy.optimize.minimize(fun = f, x0 = x0).x

def jac_h_x(k, x):
    return np.array([[-2*a*x[0], 1.0], [b, 0.0]])

def jac_o_x(k, x):
    return H + 0.5*np.diag(np.cos(x))
"""
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
    return fl.Model(dynamic_model = mc, measurement_model = om), a, b
