import autograd.numpy as np
from autograd import grad
from leapfrog import leapfrog
from toy import Simple2DGaussianMixture
import matplotlib.pyplot as plt

def computeEmpiricalBatchDistribution():
    pass
    return

def longestBatch(theta, v, eps, L, gradU=None, U=None, pi=None, visited=None):
    """
    Computation of longest batch. At least one argument among gradU, U and pi must be given (preferably gradU)
    :param theta: initial position
    :param v: initial momentum
    :param eps: step size
    :param L: number of steps in whose state we are interested (the iteration continues until an U-turn)
    :param gradU: gradient of the potential function U. Defaults to None.
    :param U: potential function U. Defaults to None.
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :param visited: list to add each visited node (we do nothing if it is None)
    :return: tuple (theta,v,l) where l is the longest batch and theta, v are the state in the Lth path (if l >= L,
             otherwise the last visited state; this is for avoiding to restart leapfrog when calling this method)
    """
    l = 0
    theta_, v_ = theta.copy(), v.copy()
    theta_L, v_L = None, None
    while np.sum((theta_ - theta) * v_) >= 0:  # while no U-turn
        l += 1
        theta_, v_ = leapfrog(theta_, v_, eps, 1, gradU, U, pi, visited)
        if l == L:
            theta_L, v_L = theta_.copy(), v_.copy()
    # return longest batch with state at step L if l >= L, else longest batch with last visited state
    return (theta_L, v_L, l) if l >= L else (theta_, v_, l)


if __name__=="__main__":
    # test code
    toy = Simple2DGaussianMixture()
    gradU = grad(toy.U)
    theta0 = np.array([0, -0.5])
    v0 = np.array([0, -1])
    theta1 = np.array([0.5, 0])
    v1 = np.array([0, 1])
    theta2 = np.array([0.5, -0.5])
    v2 = np.array([2, 1])
    eps = 0.1

    # test longestBatch
    visited = [(theta0, v0)]
    _, _, l = longestBatch(theta0, v0, eps, int(4 * np.pi / eps), gradU=gradU, visited=visited)
    print("l =", l)
    lfpathx0 = [x[0][0] for x in visited]
    lfpathy0 = [x[0][1] for x in visited]
    visited = [(theta1, v1)]
    _, _, l = longestBatch(theta1, v1, eps, int(4 * np.pi / eps), gradU=gradU, visited=visited)
    print("l =", l)
    lfpathx1 = [x[0][0] for x in visited]
    lfpathy1 = [x[0][1] for x in visited]
    visited = [(theta2, v2)]
    _, _, l = longestBatch(theta2, v2, eps, int(4 * np.pi / eps), gradU=gradU, visited=visited)
    print("l =", l)
    lfpathx2 = [x[0][0] for x in visited]
    lfpathy2 = [x[0][1] for x in visited]

    grid_lim = 3
    nb_points = 100
    xplot = np.linspace(-grid_lim, grid_lim, nb_points)
    yplot = np.linspace(-grid_lim, grid_lim, nb_points)
    X, Y = np.meshgrid(xplot, yplot)
    s = np.dstack((X, Y))
    Z = np.array([[toy.pi(s[i, j]) for j in range(nb_points)] for i in range(nb_points)]).squeeze()

    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    plt.plot(lfpathx0, lfpathy0, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.plot(lfpathx1, lfpathy1, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.plot(lfpathx2, lfpathy2, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.show()