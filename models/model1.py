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
A 2D problem with known solution
"""
# Some useful constants for defining the problem
d = 2
zero = np.zeros(d)
id = np.identity(d)

# Define the dynamic model
cov_h = id
prior = sm.Simulation(target_rv = sm.RVContinuous(name = 'normal', mean = zero, cov = id), algorithm = lambda *args: np.random.multivariate_normal(zero, id))
A = np.array([[1.0, 1.5],[0, 1.0]])
func_h = lambda k, x, noise: np.dot(A, x) + noise
#noise_sim_h = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mean = zero, cov = cov_h))
#conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero), cov = cov_h)

# Define the observation model
cov_o = 0.01*id
H = np.array([[1.0, 1.0],[0.0, 2.0]])
func_o = lambda k, x, noise: np.dot(H, x) + noise
#noise_sim_o = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mean = zero, cov = cov_o))
#conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(k, condition, zero), cov = cov_o)

# creates a Model object to feed the filter / combine the models
def model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = cov_h)#, noise_sim = noise_sim_h,  conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = cov_o)#, noise_sim = noise_sim_o,  conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om)

cov_h_i = np.linalg.inv(cov_h)
cov_o_i = np.linalg.inv(cov_o)

# Define F and compute its minimum and gradient
def F(k, x, x_prev, observation):
    a = x - np.dot(A, x_prev)
    b = observation - np.dot(H, x)
    return 0.5*(np.linalg.multi_dot([a.T, cov_h_i, a]) + np.linalg.multi_dot([b.T, cov_o_i, b]))

L = cov_h_i.T + np.linalg.multi_dot([H.T, cov_o_i, H])
P = np.dot(cov_h_i.T, A)
Q = np.dot(H.T, cov_o_i.T)

def argmin_F(k, x_prev, observation):
    return np.linalg.solve(L, np.dot(P, x_prev) + np.dot(Q, observation))

def grad_F(k, x, x_prev, observation):
    a = x - np.dot(A, x_prev)
    b = observation - np.dot(H, x)
    return np.dot(a.T, cov_h_i) - np.linalg.multi_dot([b.T, cov_o_i, H])

hess_F = cov_h_i + np.linalg.multi_dot([H.T, cov_o_i, H])
def hess_F(k, x, x_prev, observation):
    return hess

def jac_h_x(k, x):
    return A

def jac_h_n(k, x):
    return id

def jac_o_x(h, x):
    return H

def jac_o_n(k, x):
    return id


"""
Exact solution to the filtering problem
"""
def one_step_predict_update(m, P, y):
    # predict mean and covariance
    m_ = np.dot(A, m)
    P_ = np.linalg.multi_dot([A, P, A.T]) + cov_h# <-- mc.sigma
    # update mean and covariance
    v = y - np.dot(H, m_)
    S = np.linalg.multi_dot([H, P_, H.T]) + cov_o
    K = np.linalg.multi_dot([P_, H.T, np.linalg.inv(S)])
    #print("\n~~~~~~~~~~~~~ K = \n{}\n{}\n{}\n{}\n ~~~~~~~~~~~~~~~~\n".format(K, P, S, om.sigma))
    m_ += np.dot(K,v)
    P_ -= np.linalg.multi_dot([K, S, K.T])
    return m_, P_

def update(Y, m0 = zero, P0 = id):
    m, P = m0, P0
    means = [m]
    covs = [P]
    for y in Y:
        m, P =  one_step_predict_update(m, P, y)
        means.append(m)
        covs.append(P)
    return np.array(means), np.array(covs)
