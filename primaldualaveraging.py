import autograd.numpy as np
from leapfrog import leapfrog
import scipy.stats as st


def findReasonableEpsilon(theta, U):
    '''
    Heuristics for choosing an initial value of epsilon BEFORE tuning
    -- Remark as in leapflog file, we suppose at this version that M = Identity matrix---
    :return: eps
    '''
    eps = 1
    n = len(theta)
    #mean, cov = np.zeros(n), numpy.eye(n)
    #v = np.random.multivariate_normal(mean, cov)
    v = st.norm(0, 1).rvs(n)
    log_prob = -U(theta) - np.dot(v, v) / 2
    tilde_log_prob = -U(tilde_theta) - np.dot(tilde_v, tilde_v) / 2
    condition = tilde_log_prob - log_prob + log(2)
    a = 2 * (condition >0) - 1
    while a * condition > 0:
        eps = 2**a * eps
        tilde_theta, tilde_v = leapfrog(theta=theta, v=v, eps=eps, L=1, U=U)
        tilde_log_prob = -U(tilde_theta) - np.dot(tilde_v, tilde_v) / 2
        condition = tilde_log_prob - log_prob + log(2)
    return eps


def tuneEpsilon():
    pass
    return

if __name__=="__main__":
    # test code
    pass
