import autograd.numpy as np
import matplotlib.pyplot as plt
from evalmetrics import KS
from toy import MultivariateNormalDistribution, MultivariateNormalDistributionTensorflowVersion
import time
import tensorflow as tf
import tensorflow_probability as tfp


NOW = 1639196754

# some constants
lambd = 2*np.pi  # target simulation length for dual averaging, lambda = L*eps where L is leapfrog's nb of steps
deltas = np.arange(0.6, 1., 0.05)  # target acceptance probs for dual averaging; same as Wu, Stoehr, Robert paper
n_chains = 10
n_tunning_eps = 500  # n iterations for tunning epsilon
n_emp_batch_distr = 200  # n iterations for computing the empirical batch distribution
n_iter = 3000  # n samples for the MCMC

bins = np.linspace(-2,2,100)
n_iter_ground_truth = 25000
n_burnin_ground_truth = 2000

toy = MultivariateNormalDistribution()
toy_tf = MultivariateNormalDistributionTensorflowVersion()

# recover saved data
eps_opt_list = np.load('./simu_output/eps_opt_list_{}.npy'.format(NOW))
NUTS_out_arr = np.load('./simu_output/NUTS_{}-chains_{}-iters_{}.npy'.format(n_chains,n_iter,NOW))
eHMC_out_arr = np.load('./simu_output/eHMC_{}-chains_{}-iters_{}.npy'.format(n_chains,n_iter,NOW))
reference = np.load('./simu_output/Groundtruth_KS_{}_iters.npy'.format(n_iter_ground_truth))
# reference has shape (dims, bins)

# NUTS
CDF_NUTS = []  # np.array(CDF_NUTS) will have shape (len(deltas), n_chains, toy_tf.dim, bins)
for experiments in NUTS_out_arr:
    # each experiment has shape (n_chains, n_iter, toy_tf.dim)
    l1 = []
    for chain in experiments:
        l2 = []
        for d in range(chain.shape[1]):
            projected = chain[:,d]
            hist, _ = np.histogram(projected, bins)
            hist = hist / sum(hist)
            l2.append(np.cumsum(hist))
        l1.append(l2)
    CDF_NUTS.append(l1)

# eHMC
CDF_eHMC = []  # np.array(CDF_eHMC) will have shape (len(deltas), n_chains, toy_tf.dim, bins)
for experiments in eHMC_out_arr:
    # each experiment has shape (n_chains, n_iter, toy_tf.dim)
    l1 = []
    for chain in experiments:
        l2 = []
        for d in range(chain.shape[1]):
            projected = chain[:,d]
            hist, _ = np.histogram(projected, bins)
            hist = hist / sum(hist)
            l2.append(np.cumsum(hist))
        l1.append(l2)
    CDF_eHMC.append(l1)

# KS (actually, the maximum KS taken across all the dimensions)
KS_NUTS = []  # np.array(KS_NUTS) will have shape (len(deltas), n_chains)
for experiments in CDF_NUTS:
    l1 = []
    for chain in experiments:
        l2 = []
        for cdf,ref in zip(chain, reference):
            l2.append(KS(cdf, ref))
        l1.append(max(l2))
    KS_NUTS.append(l1)

KS_eHMC = []  # np.array(KS_eHMC) will have shape (len(deltas), n_chains)
for experiments in CDF_eHMC:
    l1 = []
    for chain in experiments:
        l2 = []
        for cdf, ref in zip(chain, reference):
            l2.append(KS(cdf, ref))
        l1.append(max(l2))
    KS_eHMC.append(l1)

plt.figure()
plt.boxplot([KS_eHMC[4], KS_NUTS[4]], labels=['eHMC', 'NUTS'])  # 4 is the position for delta=0.8
plt.title('KS distance (max across dims) for $\delta$=0.8')
plt.show()

