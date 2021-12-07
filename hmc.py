import matplotlib.pyplot as plt

from leapfrog import leapfrog
from utils import H
from toy import Simple2DGaussianMixture
from autograd import grad
import autograd.numpy as np
import time


def stantardHMC(theta_0, eps, L, N, M=None, U=None, pi=None, return_vs=False):
    '''
    :param theta_0: starting position (1D, otherwise we will flatten)
    :param eps: step sizer epsilon
    :param L: number of leapfrog steps
    :param N: number of HMC iterations
    :param M: covariance matrice
    :param U: potential function U
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U))
    :param return_vs: whether to return the visited v's (in a separate list) or not. Defaults to False.
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
        v = np.random.multivariate_normal(np.zeros(theta_0.shape), M)
        theta_star, v_star = leapfrog(thetas[-1], v, eps, L, M, gradU, U, pi)
        rho = np.exp(H(thetas[-1], v, U, Minv) - H(theta_star, -v_star, U, Minv))
        event = np.random.uniform(0, 1)
        if event <= rho:
            thetas.append(theta_star)
            if return_vs:
                momentums.append(-v_star)
        else:
            thetas.append(thetas[-1])
            if return_vs:
                momentums.append(v)

    return thetas if not return_vs else (tehtas, momentums)


if __name__ == "__main__":
    # test code
    np.random.seed(42)
    M = np.eye(2)
    theta0 = np.array([0, -0.5])
    eps = 0.1
    L = int(4 * np.pi / eps)
    N = 10

    # TODO
    toy = Simple2DGaussianMixture()
    gradU = grad(toy.U)
    positions = stantardHMC(theta0, eps, L, N, M, U=toy.U)

    lfpathx = [el[0] for el in positions]
    lfpathy = [el[1] for el in positions]

    '''step = np.linspace(0, N, N+1)
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(lfpathx, lfpathy, c=np.tan(step), marker='_')
    plt.show()'''

    grid_lim = 4
    nb_points = N
    xplot = np.linspace(-grid_lim, grid_lim, nb_points)
    yplot = np.linspace(-grid_lim, grid_lim, nb_points)
    X, Y = np.meshgrid(xplot, yplot)
    s = np.dstack((X, Y))
    Z = np.array([[toy.pi(s[i, j]) for j in range(nb_points)] for i in range(nb_points)]).squeeze()
    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    plt.plot(lfpathx, lfpathy, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.show()

