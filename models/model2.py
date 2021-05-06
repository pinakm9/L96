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
A 1D non-linear problem
"""
two, zero = np.array([2.0]), np.array([0.0]),
# Define the dynamic model
x_0 = np.array([1.0])
alpha, theta, w  = 3, 0.5, 0.04
sigma_h = 0.1#alpha*theta**2
prior = sm.Simulation(algorithm = lambda *args: x_0)
process_noise = sm.Simulation(algorithm = lambda* args: np.array([np.random.gamma(shape = alpha, scale = theta)]))
func_h = lambda k, x, noise: np.array([1.0 + np.sin(w*np.pi*k)]) + 0.5*x + noise
conditional_pdf_h = lambda k, x, past: scipy.stats.gamma.pdf(x[0], a = alpha, loc = func_h(k, past, zero)[0], scale = theta)

# Define the observation model
sigma_o, threshold = 0.0001, 5#int(size/2)
observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal([0.0], [[sigma_o]]))
f1 = lambda x, noise: 0.2*x**2 + noise
f2 = lambda x, noise: 0.5*x + noise - two
def func_o(k, x, noise):
    #print('k = {}'.format(k))
    if k < threshold:
        return f1(x, noise)
    else:
        return f2(x, noise)

def conditional_pdf_o(k, y, condition):
    if k < threshold:
        prob = scipy.stats.multivariate_normal.pdf(y, mean = f1(condition, zero), cov = [[sigma_o]])
    else:
        prob = scipy.stats.multivariate_normal.pdf(y, mean = f2(condition, zero), cov =  [[sigma_o]])
    #print('k = {}, y = {}, condition = {}, f1 = {}, f2 = {}, prob = {}'.format(k, y, condition, f1(condition, zero), f2(condition, zero), prob))
    return prob

# Define F and compute its minimum and gradient
def F(k, x, x_prev, observation):
    a = x - func_h(k, x_prev, zero)
    b = observation - func_o(k, x, zero)
    return  (0.5*(a**2/sigma_h + b**2/sigma_o))[0] # (a/theta - (alpha - 1.0)*np.log(a) + 0.5*b**2/sigma_o)[0]

def grad_F(k, x, x_prev, observation):
    a = x - func_h(k, x_prev, zero)
    b = observation - func_o(k, x, zero)
    if k < threshold:
        return  a/sigma_h - 0.4*x[0]*b/sigma_o # np.array([1.0/theta - (alpha - 1.0)/a[0] - (b[0]/sigma_o)*(0.4*x[0])])
    else:
        return a/sigma_h - 0.5*b/sigma_o # np.array([1.0/theta - (alpha - 1.0)/a[0] - (b[0]/sigma_o)*0.5])

def argmin_F(k, x_prev, observation):
    f = lambda x: F(k, x, x_prev, observation)
    if k < threshold:
        a = 0.2/sigma_o
        b = 1.0/sigma_h
        c = -(func_h(k, x_prev, observation)/sigma_h + observation/sigma_o)
        x0 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        return  np.array([scipy.optimize.minimize(fun = f, x0 = x0).x]) # #np.array([scipy.optimize.minimize(fun = f, x0 = theta*func_h(k, x_prev, observation)[0] - alpha + 1.0).x])
    else:
        b = (1.0/sigma_h + 0.25/sigma_o)
        c = (func_h(k, x_prev, observation)/sigma_h + 0.5*(observation + 2.0)/sigma_o)
        return c/b


# creates a Model object to feed the filter / combine the models
def model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = np.array([[sigma_h]]))#, noise_sim = process_noise, conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = np.array([[sigma_o]]), noise_sim = observation_noise, conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om)
