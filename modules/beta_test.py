import numpy as np 
import gekko as gk 

m = gk.GEKKO()
m.options.SOLVER = 3 
m.solver_options = ['linear_solver ma97'] 


x = m.Var(value=1,lb=1,ub=5)
n=50
y = 1.
s = 0.5
z = y + np.random.normal(scale=s, size=n)
alpha = 0.5 * (y - z)**2 / s**2

def func(x):
    term_1, term_2 = 0., 0.
    x = np.arange(0.01, 1., 1./100.)
    for i in range(n):
        p = np.exp(-alpha[i]/x)
        term_1 += p
        term_2 += p**2
    return (8. - term_1**2/term_2)

f = ()

print(f)

