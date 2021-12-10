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

def prHMC(theta_0, eps, emp_L, eta, N, M=None,U=None, pi=None, visited=None):
    '''
    :param theta_0: starting position (1D, otherwise we will flatten)
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
        M = np.eye(len(theta_0))

    theta_0 = theta_0.flatten()
    v_0 = np.random.multivariate_normal(np.zeros(len(theta_0)), M)
    w_ = (theta_0, v_0)
    thetas = [theta_0]
    momentums = []

    sigma=1
    i=1
    l=1
    Minv = np.linalg.inv(M)

    for n in range(1, N):
        L= math.ceil(numpy.random.choice(emp_L, size=1)[0]/3)
        u = np.random.uniform(0, 1)
        if u < eta:
            v = np.random.multivariate_normal(np.zeros(len(theta_0)), M)
            w_= (np.transpose((thetas[-1], v)), LFpath(thetas[-1], v, eps, L, M, gradU, U, pi, visited)) #TODO
            #print(LFpath(thetas[-1], v, eps, L, M, gradU, U, pi, visited))
            l=L+1
            rho =  #TODO yasmine: i don't understand how to compute this rho
            event = np.random.uniform(0, 1)
            if event <= rho:
                #TODO what are the indices of w+ in algo?
                theta, v, i, sigma = (, , , 1)
            else:
                theta, v, i, sigma = (thetas[-1], -v, 1, -1)
        else:
            j = i + sigma*L
            if sigma == 1:
                delta = j-l
            if sigma == -1:
                delta = 1-j

            if delta > 0:
                l=l+delta
                thetas_vs_star = []
                thetas_minus_vs_star = []
                (theta_star, v_star) = LFpath(thetas[-1], v, eps, delta, M, gradU, U, pi, visited)
                thetas_vs_star.append((theta_star, v_star))
                thetas_minus_vs_star.append((theta_star, -v_star))

                if sigma==1:
                    thetas_vs_star_T = list(np.transpose(thetas_vs_star))
                    w_ = [w_] + thetas_vs_star_T
                else:
                    last_to_first = [x for x in thetas_minus_vs_star[::-1]]
                    thetas_minus_vs__star_T = list(np.transpose(last_to_first))
                    w_ = thetas_minus_vs__star_T + [w_]
                    i,j = i+delta, 1

            rho =  # TODO yasmine: i don't understand how to compute this rho
            event = np.random.uniform(0, 1)
            if event <= rho:
                # TODO what are the indices of w+ in algo?
                theta, v, i, sigma = (, , , sigma)
            else:
                theta, v, i, sigma = (thetas[-1], -v, i, -sigma)

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
