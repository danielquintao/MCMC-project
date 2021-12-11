from leapfrog import leapfrog
from longestbatch import computeEmpiricalBatchDistribution
from utils import H

import matplotlib.pyplot as plt

from toy import Simple2DGaussianMixture
from autograd import grad
import autograd.numpy as np
import numpy
import time
import math


# prHMC

def prHMC(theta, eps, emp_L, eta, N, M=None, U=None, pi=None):
    '''
    :param theta: starting position (1D, otherwise we will flatten)
    :param eps: step size
    :param emp_L: empirical distribution of longest batches
    :param eta: refreshment probability
    :param N: number of eHMC iterations
    :param M: covariance matrice
    :param U: potential function U. Defaults to None.
    :param H: hamiltonian function H
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :param visited: list to add each visited node
    :return: list of accepted positions
    '''

    if U is None and pi is None:
        raise ValueError("U or pi must be given")
    if U is None:
        U = lambda x: -np.log(pi(x))
    gradU = grad(U)

    if M is None:
        M = np.eye(len(theta))

    theta = theta.flatten()
    v_0 = np.random.multivariate_normal(np.zeros(len(theta)), M)
    w_ = [(theta, v_0)]
    momentums = []

    sigma = 1
    i = 1
    l = 1
    Minv = np.linalg.inv(M)
    thetas = []

    for n in range(1, N):
        L = math.ceil(numpy.random.choice(emp_L, size=1)[0] / 3)
        u = np.random.uniform(0, 1)
        if u < eta:
            v = np.random.multivariate_normal(np.zeros(len(theta)), M)
            w_ = [(theta, v)]
            w_.extend(LFpath(theta, v, eps, L, M, gradU, U, pi))

            l = L + 1
            rho = np.exp(H(theta, v, U, Minv) - H(w_[-1][0], w_[-1][1], U, Minv))
            event = np.random.uniform(0, 1)
            if event <= rho:
                theta, v, i, sigma = (w_[-1][0], w_[-1][1], l, 1)
            else:
                theta, v, i, sigma = (theta, -v, 1, -1)
            thetas.append(theta)
        else:
            j = i + sigma * L

            if sigma == 1:
                delta = j - l
            if sigma == -1:
                delta = 1 - j

            if delta > 0:

                l = l + delta
                thetas_minus_vs_star = []
                thetas_vs_star = LFpath(theta, v, eps, delta, M, gradU, U, pi)
                for m in range(delta - 1, -1, -1):
                    thetas_minus_vs_star.append((thetas_vs_star[m][0], -thetas_vs_star[m][1]))

                if sigma == 1:
                    w_.extend(thetas_vs_star)
                else:
                    thetas_minus_vs_star.extend(w_)
                    w_ = thetas_minus_vs_star
                    i, j = i + delta, 1

            rho = np.exp(H(theta, v, U, Minv) - H(w_[j - 1][0], -sigma * w_[j - 1][1], U, Minv))
            event = np.random.uniform(0, 1)
            if event <= rho:
                theta, v, i, sigma = (w_[j - 1][0], -sigma * w_[j - 1][1], j, sigma)
            else:
                theta, v, i, sigma = (theta, -v, i, -sigma)
            thetas.append(theta)
    return thetas


### adding this solely for commit problem test

def LFpath(theta, v, eps, L, M=None, gradU=None, U=None, pi=None):
    """
    Leapfrog integration. At least one argument among gradU, U and pi must be given (preferably gradU)
    :param theta: initial position (1D, otherwise we will flatten)
    :param v: initial momentum (1D, otherwise we will flatten)
    :param eps: step size
    :param L: number of steps
    :param M: covariance matrice
    :param gradU: gradient of the potential function U. Defaults to None.
    :param U: potential function U. Defaults to None.
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :return: tuple (theta,v) obtained with L steps of the integrator
    """
    theta = theta.flatten()
    v = v.flatten()
    # gradient of U
    if gradU is None and U is None and pi is None:
        raise ValueError("At least one among gradU, U and pi should be given")
    if gradU is None:
        U = (lambda x: -np.log(pi(x))) if U is None else U
        gradU = grad(U)
    if M is None:
        M = np.eye(len(theta))
    invM = np.linalg.inv(M)
    visited = []
    # LF
    v_star= v- eps / 2 * gradU(theta)
    for _ in range(L):
        theta = theta + eps * (invM @ v.reshape(-1, 1)).flatten()
        v = v_star - eps / 2 * gradU(theta)
        v_star = v - eps / 2 * gradU(theta)
        visited.append((theta, v))
    return visited
