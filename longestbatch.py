import autograd.numpy as np
from autograd import grad
from leapfrog import leapfrog
from utils import H, log_accept_proba # TODO -- use log_accept_proba
from toy import Simple2DGaussianMixture
import matplotlib.pyplot as plt

def computeEmpiricalBatchDistribution(theta, eps, L, K, M=None, U=None, pi=None, visited=None):
    """
    Empirical distribution of longest batches. You must provide U or pi.
    :param theta: initial position
    :param eps: step size
    :param L: number of steps for the leapfrog integrator
    :param K: number of iterations
    :param M: covariance matrice
    :param U: potential function U. Defaults to None.
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :param visited: list of lists with longestBatch path for each visited state (we do nothing if it is None).
    :return: list with the longest batch at each visited point (empirical distribution of longest batches)
    """
    if U is None and pi is None:
        raise ValueError("U or pi must be given")
    if U is None:
        U = lambda x: -np.log(pi(x))
    gradU = grad(U)

    if M is None:
        M = np.eye(len(theta))
    Minv = np.linalg.inv(M)

    empdistr = []
    dim = len(theta)
    for _ in range(K):
        # compute candidate state and longest batch for current state:
        v = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
        visited_temp = [(theta, v)] if visited is not None else None
        theta_L, v_L, l = longestBatch(theta, v, eps, L, M, gradU, visited=visited_temp)
        if l < L:  # we still need to walk some steps
            theta_L, v_L = leapfrog(theta_L, v_L, eps, L-l, M, gradU)
        # accept or reject update:
        u = np.random.rand()
        if np.log(u) < H(theta, v, U, Minv) - H(theta_L, -v_L, U, Minv):
            # accept
            theta, v = theta_L, -v_L
        # update empirical batch distribution:
        empdistr.append(l)
        # log information:
        if visited is not None:
            visited.append(visited_temp)
    return empdistr

def longestBatch(theta, v, eps, L, M=None, gradU=None, U=None, pi=None, visited=None):
    """
    Computation of longest batch. At least one argument among gradU, U and pi must be given (preferably gradU)
    :param theta: initial position
    :param v: initial momentum
    :param eps: step size
    :param L: number of steps in whose state we are interested (the iteration continues until an U-turn)
    :param M: covariance matrice
    :param gradU: gradient of the potential function U. Defaults to None.
    :param U: potential function U. Defaults to None.
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :param visited: list to add each visited node (we do nothing if it is None)
    :return: tuple (theta,v,l) where l is the longest batch and theta, v are the state in the Lth path (if l >= L,
             otherwise the last visited state; this is for avoiding to restart leapfrog when calling this method)
    """
    if M is None:
        M = np.eye(len(theta))
    invM = np.linalg.inv(M)
    l = 0
    theta_, v_ = theta.copy(), v.copy()
    theta_L, v_L = None, None
    while np.sum((theta_ - theta) * (invM @ v_.reshape(-1,1)).flatten()) >= 0:  # while no U-turn
        l += 1
        theta_, v_ = leapfrog(theta_, v_, eps, 1, M, gradU, U, pi, visited)
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
    print(theta0, v0)
    visited = [(theta1, v1)]
    _, _, l = longestBatch(theta1, v1, eps, int(4 * np.pi / eps), gradU=gradU, visited=visited)
    print("l =", l)
    lfpathx1 = [x[0][0] for x in visited]
    lfpathy1 = [x[0][1] for x in visited]
    print(theta1, v1)
    visited = [(theta2, v2)]
    _, _, l = longestBatch(theta2, v2, eps, int(4 * np.pi / eps), gradU=gradU, visited=visited)
    print("l =", l)
    lfpathx2 = [x[0][0] for x in visited]
    lfpathy2 = [x[0][1] for x in visited]
    print(theta2, v2)

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

    # test computeEmpiricalBatchDistribution
    np.random.seed(0)

    visited = []  # now, visited will be a list of lists
    empdistr = computeEmpiricalBatchDistribution(theta0, eps, 10, 5, U=toy.U, visited=visited)
    print(empdistr, theta0)
    mainpath = [longbatch[0] for longbatch in visited]
    mainpathx = [x[0][0] for x in mainpath]
    mainpathy = [x[0][1] for x in mainpath]

    cmaps = ['Greys', 'Greens', 'Reds']
    ncmaps = len(cmaps)
    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    for i,batch in enumerate(visited):
        plt.plot([x[0][0] for x in batch], [x[0][1] for x in batch], color='k', zorder=1)
        plt.scatter([x[0][0] for x in batch], [x[0][1] for x in batch],
                 marker='o', s=10, c=np.arange(len(batch)), cmap=cmaps[i - ncmaps*(i//ncmaps)], zorder=2)
    plt.plot(mainpathx, mainpathy, marker='D', markersize=5, color='purple')
    plt.show()