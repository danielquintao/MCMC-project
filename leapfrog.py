import autograd.numpy as np
from autograd import grad
from toy import Simple2DGaussianMixture
import matplotlib.pyplot as plt


def leapfrog(theta, v, eps, L, invM=None, gradU=None, U=None, pi=None, visited=None):
    """
    Leapfrog integration. At least one argument among gradU, U and pi must be given (preferably gradU)
    :param theta: initial position (1D)
    :param v: initial momentum (1D)
    :param eps: step size
    :param L: number of steps
    :param invM: inverse of the mass matrix
    :param gradU: gradient of the potential function U. Defaults to None.
    :param U: potential function U. Defaults to None.
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :param visited: list to add each visited node (we do nothing if it is None)
    :return: tuple (theta,v) obtained with L steps of the integrator
    """
    if theta.shape != v.shape:
        raise ValueError("theta and v can be either (1,n) or (n,), but should have the same shape")
    # gradient of U
    if gradU is None and U is None and pi is None:
        raise ValueError("At least one among gradU, U and pi should be given")
    if gradU is None:
        U = (lambda x: -np.log(pi(x))) if U is None else U
        gradU = grad(U)
    if invM is None:
        invM = np.eye(len(theta))
    # Leapfrog
    for _ in range(L):
        v = v - eps / 2 * gradU(theta)
        theta = theta + eps * (invM @ v.reshape(-1,1)).flatten()
        v = v - eps / 2 * gradU(theta)
        if visited is not None:
            visited.append((theta, v))
    return theta, v


if __name__ == "__main__":
    # test code
    toy = Simple2DGaussianMixture()
    gradU = grad(toy.U)
    theta0 = np.array([0,-0.5])
    v0 = np.array([0,-1])
    eps = 0.1

    grid_lim = 3
    nb_points = 100
    xplot = np.linspace(-grid_lim, grid_lim, nb_points)
    yplot = np.linspace(-grid_lim, grid_lim, nb_points)
    X, Y = np.meshgrid(xplot, yplot)
    s = np.dstack((X, Y))
    Z = np.array([[toy.pi(s[i, j]) for j in range(nb_points)] for i in range(nb_points)]).squeeze()

    # API 1 - calling leapfrog with gradU
    visited = [(theta0, v0)]
    _ = leapfrog(theta0, v0, eps, int(4*np.pi/eps), gradU=gradU, visited=visited)
    lfpathx = [x[0][0] for x in visited]
    lfpathy = [x[0][1] for x in visited]
    print(theta0, v0)

    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    plt.plot(lfpathx, lfpathy, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.show()

    # API 2 - calling leapfrog with U
    visited = [(theta0, v0)]
    _ = leapfrog(theta0, v0, eps, int(4 * np.pi / eps), U=toy.U, visited=visited)
    lfpathx = [x[0][0] for x in visited]
    lfpathy = [x[0][1] for x in visited]
    print(theta0, v0)

    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    plt.plot(lfpathx, lfpathy, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.show()

    # API 3 - calling leapfrog with the target distr
    visited = [(theta0, v0)]
    _ = leapfrog(theta0, v0, eps, int(4 * np.pi / eps), pi=toy.pi, visited=visited)
    lfpathx = [x[0][0] for x in visited]
    lfpathy = [x[0][1] for x in visited]
    print(theta0, v0)

    plt.figure()
    plt.contourf(X, Y, Z, cmap="PuBu_r")
    plt.plot(lfpathx, lfpathy, marker='o', markersize=2, color='k', markeredgecolor='orange')
    plt.show()
