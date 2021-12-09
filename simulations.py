import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from ehmc import eHMC
from longestbatch import computeEmpiricalBatchDistribution
from primaldualaveraging import findReasonableEpsilon_hmc_dual_averaging
from evalmetrics import minESS, ESJD, KS
from toy import Simple2DGaussianMixture, LectureExample2, MultivariateNormalDistribution
import time

# replication
np.random.seed(1)

# for saving data
NOW = int(time.time())

# TODO -- normalize metrics w.r.t. gradient

# some constants
lambd = 2*np.pi  # target simulation length for dual averaging, lambda = L*eps where L is leapfrog's nb of steps
deltas = np.arange(0.6, 1., 0.05)  # target acceptance probs for dual averaging; same as Wu, Stoehr, Robert paper
n_chains = 3
n_tunning_eps = 500  # n iterations for tunning epsilon
n_emp_batch_distr = 200  # n iterations for computing the empirical batch distribution
n_iter = 2000  # n samples for the MCMC

########################################################
# Simulation: Multivariate Normal Distribution
########################################################

toy = MultivariateNormalDistribution()

# NUTS/Stan -- TODO

# eHMC
eHMC_out = [] # NOTE -- list of lists
eps_opt_list = []
print("Computing eHMC...")
for delta in deltas:
    print("\tdelta = {}".format(delta))
    # TODO question: which steps are made once for all the n_chains experiments, and which ones are repeated each time?
    # tune epsilon
    theta = np.random.rand(toy.dim)
    START = time.time()
    eps_opt = findReasonableEpsilon_hmc_dual_averaging(theta=theta, delta=delta, lambd=lambd,
                                                   n_samples=n_tunning_eps, n_samples_adap=n_tunning_eps, U=toy.U)
    print('\t(tunning epsilon ok - eps = {} - took {:.4f}s)'.format(eps_opt, time.time() - START))
    eps_opt_list.append(eps_opt)
    experiments = []
    for _ in range(n_chains):
        theta = np.random.rand(toy.dim)
        # compute the empirical longest batch distribution
        START = time.time()
        emp_batch_distr = computeEmpiricalBatchDistribution(theta=theta, eps=eps_opt, L=int(lambd/eps_opt),
                                                            K=n_emp_batch_distr, U=toy.U)
        print('\t\tempirical distr ok - took {:.4f}s'.format(time.time() - START))
        # eHMC
        START = time.time()
        samples = eHMC(theta_0=theta, eps=eps_opt, emp_L=emp_batch_distr, N=n_iter, U=toy.U)
        print('\t\tchain ok - took {:.4f}s'.format(time.time() - START))
        experiments.append(samples)
    eHMC_out.append(experiments)
np.save('./simu_output/eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(eHMC_out))

# compute metrics
minESS_per_chain_eHMC, minESS_eHMC = [], []
ESJD_per_chain_eHMC, ESJD_eHMC = [], []
# TODO -- add KS distance
print("Computing metrics for eHMC...")
compute_for_each = True  # whether to compute the metrics for each individual chain (and delta) or only for each delta
for experiments,delta in zip(eHMC_out,deltas):  # one list of experiments per delta
    print("\tdelta = {}".format(delta))
    if compute_for_each:
        for sampling in experiments:  # n_chains samplings per list of experiments
            X = np.array(sampling)  # X has shape (n_iter, dim)
            START = time.time()
            min_ess = minESS(X)
            print('\t\tminESS for single chain = {}, took {:.4f}s'.format(min_ess, time.time()-START))
            minESS_per_chain_eHMC.append(min_ess)
            START = time.time()
            esjd = ESJD(X)
            print('\t\tESJD for single chain = {}, took {:.4f}s'.format(esjd, time.time() - START))
            ESJD_per_chain_eHMC.append(esjd)
    X = np.array(experiments)
    START = time.time()
    min_ess = minESS(X)
    print('\t\tminESS = {}, took {:.4f}s'.format(min_ess, time.time() - START))
    minESS_eHMC.append(min_ess)
    START = time.time()
    esjd = ESJD(X)
    print('\t\tESJD = {}, took {:.4f}s'.format(esjd, time.time() - START))
    ESJD_eHMC.append(esjd)
np.save('./simu_output/minESS_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(minESS_eHMC))
np.save('./simu_output/minESS_PER_CHAIN_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(minESS_per_chain_eHMC))
np.save('./simu_output/ESJD_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(ESJD_eHMC))
np.save('./simu_output/ESJD_PER_CHAIN_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(ESJD_per_chain_eHMC))

# visualization
x_for_each = np.repeat(deltas, n_chains)

plt.figure()
if len(minESS_per_chain_eHMC) > 0:
    plt.scatter(x_for_each, minESS_per_chain_eHMC, color='k')
plt.plot(deltas, minESS_eHMC, 'rD-')
plt.xlabel('target acceptance rate')
plt.title('minESS, eHMC')
plt.show()

plt.figure()
if len(ESJD_per_chain_eHMC) > 0:
    plt.scatter(x_for_each, ESJD_per_chain_eHMC, color='k')
plt.plot(deltas, ESJD_eHMC, 'rD-')
plt.xlabel('target acceptance rate')
plt.title('ESJD, eHMC')
plt.show()