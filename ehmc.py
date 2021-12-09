from leapfrog import leapfrog
from longestbatch import computeEmpiricalBatchDistribution
from utils import H

import matplotlib.pyplot as plt

from toy import Simple2DGaussianMixture
from autograd import grad
import autograd.numpy as np
import numpy
import time

def eHMC(theta_0, eps, emp_L, N, M=None,U=None, pi=None, visited=None):
    '''
    :param theta_0: starting position (1D, otherwise we will flatten)
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
    theta_0 = theta_0.flatten()
    thetas = [theta_0]
    momentums = []

    if U is None and pi is None:
        raise ValueError("U or pi must be given")
    if U is None:
        U = lambda x: -np.log(pi(x))
    gradU = grad(U)

    if M is None:
        M = np.eye(len(theta_0))
    Minv = np.linalg.inv(M)

    for k in range(N):

        v=np.random.multivariate_normal(np.zeros(len(theta_0)), M)
        L= numpy.random.choice(emp_L, size=1)[0]
        theta_star, v_star = leapfrog(thetas[-1], v, eps, L, Minv, gradU, U, pi, visited)
        rho = np.exp(H(thetas[-1], v, U, Minv) - H(theta_star, -v_star, U, Minv))
        event = np.random.uniform(0, 1)
        if event <= rho:
            thetas.append(theta_star)
            momentums.append(-v_star)
        else:
            thetas.append(thetas[-1])
            momentums.append(v)

    return thetas



if __name__=="__main__":
    # test code
    np.random.seed(42)
    theta0 = np.array([0, -0.5])
    eps = 0.1
    N = 10

    # TODO
    toy = Simple2DGaussianMixture()
    gradU = grad(toy.U)

    # find emp_L using computeEmpiricalBatchDistribution
    visited = []  # now, visited will be a list of lists
    emp_L=computeEmpiricalBatchDistribution(theta0, eps, 10, 5, U=toy.U, visited=visited)

    #test eHMC
    visited1= []
    positions = eHMC(theta0, eps, emp_L, N, U=toy.U, visited=visited1)

    lfpathx = [el[0] for el in positions]
    lfpathy = [el[1] for el in positions]


    grid_lim = 4
    nb_points = 100
    xplot = np.linspace(-grid_lim, grid_lim, nb_points)
    yplot = np.linspace(-grid_lim, grid_lim, nb_points)
    X, Y = np.meshgrid(xplot, yplot)
    s = np.dstack((X, Y))
    Z = np.array([[toy.pi(s[i, j]) for j in range(nb_points)] for i in range(nb_points)]).squeeze()

    cmaps = ['Greys', 'Greens', 'Reds']
    ncmaps = len(cmaps)

    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    '''for i, batch in enumerate(visited1):
        plt.plot([x[0][0] for x in batch], [x[0][1] for x in batch], color='k', zorder=1)
        plt.scatter([x[0][0] for x in batch], [x[0][1] for x in batch],
                    marker='o', s=10, c=np.arange(len(batch)), cmap=cmaps[i - ncmaps * (i // ncmaps)], zorder=2)'''

    plt.plot(lfpathx, lfpathy, marker='D', markersize=5, color='purple', markeredgecolor='orange')
    plt.show()








