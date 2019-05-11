import numpy as np
from scipy.optimize import minimize


def steep_direct_cost(alpha, grad_disc_matrix):
	"""
	Calculates ||sum_i alpha_i grad_disc_i||^2
	- alpha: k-dim array with alpha values
	- grad_disc_matrix: list with gradients of all discriminators
					  grad_disc_matrix = NxK, where N = number of params
	"""

	v = np.sum(np.multiply(grad_disc_matrix, alpha), axis=1)

	return np.inner(v, v)


def steep_direc_cost_deriv(alpha, grad_disc_matrix=[]):

	v = np.sum(np.multiply(grad_disc_matrix, alpha), axis=1)

	deriv = 2 * np.matmul(np.transpose(v), grad_disc_matrix)

	return np.ndarray.tolist(deriv)


def make_constraints(n_disc):
	cons = [{'type': 'eq', 'fun': lambda alpha: np.array([np.sum(alpha) - 1]), 'jac': lambda alpha: np.ones([1, n_disc])}]

	bounds = []
	for k in range(n_disc):
		jacobian = np.zeros([1, n_disc])
		jacobian[0, k] = 1.

		ineq = {'type': 'ineq', 'fun': lambda alpha: np.array([alpha[k]]), 'jac': lambda alpha: jacobian}
		cons.append(ineq)

		bounds.append((0, 1.))
	return cons, bounds


if __name__ == '__main__':
	alpha = np.array([0.5, 0.25, 0.25, 0.25])
	grad_disc_matrix = np.ones([10, 4])

	const, bounds = make_constraints(4)

	print(const)

	res = minimize(steep_direct_cost, alpha, args=grad_disc_matrix, jac=steep_direc_cost_deriv, bounds=bounds, constraints=const, method='SLSQP', options={'disp': True})

	print(res.success)

	print(res.x)
