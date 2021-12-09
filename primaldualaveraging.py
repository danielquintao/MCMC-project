import autograd.numpy as np
from leapfrog import leapfrog
import scipy.stats as st
from utils import log_accept_proba

def findReasonableEpsilon(theta, U, M=None):
    '''
    Heuristics for choosing an initial value of epsilon BEFORE tuning
    :return: eps
    '''
    invM = None if M is None else np.linalg.inv(M)
    eps = 1
    n = np.size(theta)
    # v = st.norm(0, 1).rvs(n)
    mean, cov = np.zeros(n), np.eye(n)
    v = np.random.multivariate_normal(mean, cov)
    theta_new, v_new = leapfrog(theta=theta, v=v, eps=eps, L=1, invM=invM, U=U)
    log_rho = log_accept_proba(theta, v, theta_new, v_new, U, M)
    condition = log_rho + np.log(2) #use to compare acceptance probability with 0.5
    a = 2.0 * (condition >0) - 1.0
    while a * condition > 0:
        eps = (2**a) * eps
        theta_new, v_new = leapfrog(theta=theta, v=v, eps=eps, L=1, invM=invM, U=U)
        log_rho = log_accept_proba(theta, v, theta_new, v_new, U, M)
        condition = log_rho + np.log(2)
    return eps



def findReasonableEpsilon_hmc_dual_averaging(theta, delta, lambd, n_samples, n_samples_adap, U):
    """
    Algorithm 5 of Hoffman, Gelman "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
    Inputs
    ----------
    theta: a place to start sampling from.
    delta: target mean acceptance probability
    lambd: length of leapfrog integration.
    n_samples: total number of samples to return.
    n_samples_adap: number of samples (steps) to compute an adapted epsilon

    Outputs
    -------
    eps:
    #samples: samples produced by the HMC.
    #accepted: array of 0 and 1 to display which proposed moves have been accepted
    """
    initial_position = np.array(theta)

    samples = [initial_position]
    accepted = []

    ## Setting parameters step:
    eps = findReasonableEpsilon(theta, U)
    mu = np.log(10*eps)
    log_bar_eps = 0.0
    bar_H = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    ## Get samples setp:
    size = (n_samples, len(theta))

    # all momentums
    momentum = st.norm(0, 1).rvs(size)
    for idx, v0 in enumerate(momentum):
        L = max(1, int(np.floor(lambd/eps + 0.5)))
        theta_new, v_new = leapfrog(samples[-1], v0, eps, L, U=U)

        # acceptance rate
        log_rho = log_accept_proba(samples[-1], v0, theta_new, v_new, U)
        rho = np.exp(log_rho)

        # propose next sample
        if np.random.rand() < rho:
            samples.append(theta_new)
            accepted.append(True)
        else:
            samples.append(np.copy(samples[-1]))
            accepted.append(False)

        # propose new eps if it is necessary for next iteration
        m = idx+1
        if m <= n_samples_adap:
            bar_H = (1-1/(m+t0)) * bar_H + (delta - rho)/(m+t0)
            log_eps = mu - np.sqrt(m)/gamma * bar_H
            eps = np.exp(log_eps)
            # We update log_bar_eps until this step
            log_bar_eps = m**(-kappa) * log_eps + (1-m**(-kappa)) * log_bar_eps
        else:
            eps = np.exp(log_bar_eps)
    return eps

    #return np.array(samples[1:]), np.array(accepted)


if __name__=="__main__":
    from toy import Simple2DGaussianMixture, LectureExample2, MultivariateNormalDistribution
    toy = Simple2DGaussianMixture()
    toy2 = LectureExample2()
    toy3 = MultivariateNormalDistribution()

    #np.random.seed(42)
    # test code 1
    # theta = np.array([1, 1])
    # print("Test the first function: eps=", findReasonableEpsilon(theta, toy.U))
    #
    # theta = np.array([0, 0])
    # print("Test the first function: eps=", findReasonableEpsilon(theta, toy.U))
    #
    # theta = np.array([3, 3])
    # print("Test the first function: eps=", findReasonableEpsilon(theta, toy.U))
    #
    # theta = np.array([2, 20])
    # print("Test the first function: eps=", findReasonableEpsilon(theta, toy.U))
    #
    # theta = np.array([2, 20])
    # print("Test the first function: eps=", findReasonableEpsilon(theta, toy2.log_p))

    # test code 2

    ##
    print("find best eps after some n_sample_adap steps")
    theta = np.random.rand(toy3.dim)
    print(findReasonableEpsilon_hmc_dual_averaging(theta=theta, delta=0.65, lambd=5,
                                                   n_samples=5000, n_samples_adap=5000, U=toy3.U))




