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
d = 100 # dimension of the problem, must be even
zero = np.zeros(d)
id = np.identity(d)

# Define the dynamic model
cov_h = 0.05*id
prior = sm.Simulation(target_rv = sm.RVContinuous(name = 'normal', mean = zero, cov = id), algorithm = lambda *args: np.random.multivariate_normal(zero, id))
A = np.diag([0, 0.04] + [-100.0] + [-100.0]*(d-3))
eA = scipy.linalg.expm(A)
#print(eA)
func_h = lambda k, x, noise: np.dot(eA, x) + noise
#noise_sim_h = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mean = zero, cov = cov_h))
#conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero), cov = cov_h)

# Define the observation model
cov_o = 0.05**2*id
matrices = [np.array([[1.0, 0.0],[0.0, 1.0]])]*(int(d/2))
H = scipy.linalg.block_diag(*matrices)
#print('Determinant of H = {}'.format(np.linalg.det(H)))
func_o = lambda k, x, noise: np.dot(H, x) + noise
#noise_sim_o = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mean = zero, cov = cov_o))
#conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(k, condition, zero), cov = cov_o)

# creates a Model object to feed the filter / combine the models
def model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = cov_h)#, noise_sim = noise_sim_h,  conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = cov_o)#, noise_sim = noise_sim_o,  conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om), d

# Define F and compute its minimum and gradient
cov_h_i = np.linalg.inv(cov_h)
cov_o_i = np.linalg.inv(cov_o)

def F(k, x, x_prev, observation):
    a = x - np.dot(eA, x_prev)
    b = observation - np.dot(H, x)
    return 0.5*(np.linalg.multi_dot([a.T, cov_h_i, a]) + np.linalg.multi_dot([b.T, cov_o_i, b]))

L = cov_h_i.T + np.linalg.multi_dot([H.T, cov_o_i, H])
P = np.dot(cov_h_i.T, eA)
Q = np.dot(H.T, cov_o_i.T)

def argmin_F(k, x_prev, observation):
    return np.linalg.solve(L, np.dot(P, x_prev) + np.dot(Q, observation))

def grad_F(k, x, x_prev, observation):
    a = x - np.dot(eA, x_prev)
    b = observation - np.dot(H, x)
    return np.dot(a.T, cov_h_i) - np.linalg.multi_dot([b.T, cov_o_i, H])

hess = cov_h_i + np.linalg.multi_dot([H.T, cov_o_i, H])
def hess_F(k, x, x_prev, observation):
    return hess

def jac_h_x(k, x):
    return eA

def jac_h_n(k, x):
    return id

def jac_o_x(h, x):
    return H

def jac_o_n(k, x):
    return id


# Define projected model
projection_matrix = np.zeros((d, 2))
projection_matrix[0][0] = 1.0
projection_matrix[1][1] = 1.0


#print(np.dot(projection_matrix, projection_matrix.T))

def proj_model(size):
    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = cov_h)#, noise_sim = noise_sim_h,  conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = cov_o)#, noise_sim = noise_sim_o,  conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om, projection_matrix = projection_matrix), d

# Define projected F and compute its minimum and gradient
H_ = np.dot(H.T, np.linalg.inv(np.dot(H, H.T)))
proj_H = np.linalg.multi_dot([projection_matrix.T, H_, H])
proj_cov_o = np.linalg.multi_dot([projection_matrix.T, H_, cov_o, H_.T, projection_matrix])
proj_cov_o_i = np.linalg.inv(proj_cov_o)

def proj_F(k, x, x_prev, observation):
    a = x - np.dot(eA, x_prev)
    b = observation - np.dot(proj_H, x)
    return 0.5*(np.linalg.multi_dot([a.T, cov_h_i, a]) + np.linalg.multi_dot([b.T, proj_cov_o_i, b]))

proj_L = cov_h_i.T + np.linalg.multi_dot([proj_H.T, proj_cov_o_i, proj_H])
proj_P = np.dot(cov_h_i.T, eA)
proj_Q = np.dot(proj_H.T, proj_cov_o_i.T)

def proj_argmin_F(k, x_prev, observation):
    return np.linalg.solve(proj_L, np.dot(proj_P, x_prev) + np.dot(proj_Q, observation))

def proj_grad_F(k, x, x_prev, observation):
    a = x - np.dot(eA, x_prev)
    b = observation - np.dot(proj_H, x)
    return np.dot(a.T, cov_h_i) - np.linalg.multi_dot([b.T, proj_cov_o_i, proj_H])

proj_hess = cov_h_i + np.linalg.multi_dot([proj_H.T, proj_cov_o_i, proj_H])
def proj_hess_F(k, x, x_prev, observation):
    return proj_hess

def proj_jac_h_x(k, x):
    return eA

def proj_jac_h_n(k, x):
    return id

def proj_jac_o_x(h, x):
    return proj_H

def proj_jac_o_n(k, x):
    return id

def proj_obeserved_path(observed_path):
    return np.array([np.dot(proj_H, y) for y in observed_path])
"""
Exact solution to the filtering problem
"""
def one_step_predict_update(m, P, y):
    # predict mean and covariance
    m_ = np.dot(eA, m)
    P_ = np.linalg.multi_dot([eA, P, eA.T]) + cov_h# <-- mc.sigma
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

# exact solution to the projected problem
def proj_one_step_predict_update(m, P, y):
    # predict mean and covariance
    m_ = np.dot(eA, m)
    P_ = np.linalg.multi_dot([eA, P, eA.T]) + cov_h# <-- mc.sigma
    # update mean and covariance
    v = y - np.dot(proj_H, m_)
    S = np.linalg.multi_dot([proj_H, P_, proj_H.T]) + proj_cov_o
    K = np.linalg.multi_dot([P_, proj_H.T, np.linalg.inv(S)])

    #print("\n~~~~~~~~~~~~~ K = \n{}\n{}\n{}\n{}\n ~~~~~~~~~~~~~~~~\n".format(K, P, S, om.sigma))
    m_ += np.dot(K,v)
    P_ -= np.linalg.multi_dot([K, S, K.T])
    return m_, P_

def proj_update(Y, m0 = zero, P0 = id):
    m, P = m0, P0
    means = [m]
    covs = [P]
    for y in Y:
        m, P =  proj_one_step_predict_update(m, P, y)
        means.append(m)
        covs.append(P)
    return np.array(means), np.array(covs)
