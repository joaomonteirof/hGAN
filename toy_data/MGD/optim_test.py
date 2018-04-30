import numpy as np
from scipy.optimize import minimize

def steep_direct(alpha, grad_disc_list = []):
	"""
	Calculates ||sum_i alpha_i grad_disc_i||^2
	- alpha: k-dim array with alpha values
	- grad_disc_list: list with gradients of all discriminators
					  grad_disc_list[i] = Nx1-dim vector, where N = number of params	  
	"""

	n_disc = len(grad_disc_list)
	v = 0

	for k in range(n_disc):
		v += alpha[k] * grad_disc_list[k]

	return (np.matmul(np.transpose(v), v))


def steep_direc_deriv(alpha, grad_disc_list = []):

	n_disc = len(grad_disc_list)
	v = 0

	for k in range(n_disc):
		v += alpha[k] * grad_disc_list[k]

	derivatives = []

	for k in range(n_disc):
		deriv = 2 * np.matmul(np.transpose(v), grad_disc_list[k])
		derivatives.append(deriv)

	return derivatives

cons = ({'type': 'eq',
		'fun' : lambda alpha: np.array([alpha[0] + alpha[1] - 1]),
		'jac' : lambda alpha: np.array([1., 1.])},
		{'type': 'ineq',
		'fun' : lambda alpha: np.array([alpha[0]]),
		'jac' : lambda alpha: np.array([1., 0.])},
		{'type': 'ineq',
		'fun' : lambda alpha: np.array([alpha[1]]),
		'jac' : lambda alpha: np.array([0., 1.])})
	

alpha = np.array([1, 2])
grad_disc_list = [np.ones([10, 1]), np.ones([10, 1])]


res = minimize(steep_direct, alpha, args = grad_disc_list, jac = steep_direc_deriv, constraints = cons, method = 'SLSQP', options = {'disp': True})

