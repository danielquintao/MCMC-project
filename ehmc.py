from leapfrog import leapfrog
from longestbatch import computeEmpiricalBatchDistribution

import matplotlib.pyplot as plt

from toy import Simple2DGaussianMixture
from autograd import grad
import autograd.numpy as np
import numpy
import time

def eHMC(theta_0, eps, emp_L, N, M, U=None, pi=None, visited=None):
    '''
    :param theta_0: starting position
    :param eps: step size
    :param emp_L: empirical distribution of longest batches
    :param N: number of eHMC iterations
    :param M: covariance matrice
    :param U: potential function U. Defaults to None.
    :param H: hamiltonian function H
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :param visited: list to add each visited node
    :return: list of accepted positions
    '''

    start = time.time()
    thetas = [theta_0.reshape(1, -1)]
    momentums = []

    if U is None and pi is None:
        raise ValueError("U or pi must be given")
    if U is None:
        U = lambda x: -np.log(pi(x))
    gradU = grad(U)

    # define hamiltonian
    def H(theta, v):
        return U(theta) + 0.5 * np.sum(v * v)

    for k in range(N):
        v = np.random.multivariate_normal(np.zeros(2), M, 1)
        L= numpy.random.choice(emp_L, size=1)
        theta_star, v_star = leapfrog(thetas[-1], v, eps, L, gradU, U, pi, visited)
        rho = np.exp(H(thetas[-1], v) - H(theta_star, v_star))
        event = np.random.uniform(0, 1)
        if event <= rho:
            thetas.append(theta_star)
            momentums.append(-v_star)
        else:
            thetas.append(thetas[-1])
            momentums.append(v)

    print(time.time() - start)
    return thetas



if __name__=="__main__":
    # test code
    pass