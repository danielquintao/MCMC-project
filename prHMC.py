


def eHMC(theta_0, eps, emp_L, eta, N, M=None,U=None, pi=None, visited=None):
    '''
    :param theta_0: starting position
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

    return

def LFpath(theta, v, eps, L, M=None, gradU=None, U=None, pi=None):
    """
    Leapfrog integration. At least one argument among gradU, U and pi must be given (preferably gradU)
    :param theta: initial position (1D)
    :param v: initial momentum (1D)
    :param eps: step size
    :param L: number of steps
    :param M: covariance matrice
    :param gradU: gradient of the potential function U. Defaults to None.
    :param U: potential function U. Defaults to None.
    :param pi: target distribution up to a multiplicative constant (pi is prop. to exp(-U)). Defaults to None.
    :return: tuple (theta,v) obtained with L steps of the integrator
    """
    # gradient of U
    if gradU is None and U is None and pi is None:
        raise ValueError("At least one among gradU, U and pi should be given")
    if gradU is None:
        U = (lambda x: -np.log(pi(x))) if U is None else U
        gradU = grad(U)
    if M is None:
        M = np.eye(len(theta))
    invM = np.linalg.inv(M)
    visited = [(theta, v)]
    # LF
    for _ in range(L):
        # TODO
        visited.append((theta, v))
    return visited