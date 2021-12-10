import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from ehmc import eHMC
from longestbatch import computeEmpiricalBatchDistribution
from primaldualaveraging import findReasonableEpsilon_hmc_dual_averaging
from evalmetrics import minESS, ESJD, KS
from toy import MultivariateNormalDistribution, MultivariateNormalDistributionTensorflowVersion
import time
import tensorflow as tf
import tensorflow_probability as tfp

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

tfd = tfp.distributions

# replication
np.random.seed(1)

# for saving data
NOW = int(time.time())

# TODO -- normalize metrics w.r.t. gradient

# some constants
lambd = 2*np.pi  # target simulation length for dual averaging, lambda = L*eps where L is leapfrog's nb of steps
deltas = np.arange(0.6, 1., 0.05)  # target acceptance probs for dual averaging; same as Wu, Stoehr, Robert paper
n_chains = 5
n_tunning_eps = 500  # n iterations for tunning epsilon
n_emp_batch_distr = 200  # n iterations for computing the empirical batch distribution
n_iter = 2000  # n samples for the MCMC

########################################################
# Simulation: Multivariate Normal Distribution
########################################################

toy = MultivariateNormalDistribution()
toy_tf = MultivariateNormalDistributionTensorflowVersion()

# Dual Averaging
nuts_kernel_for_dual_averaging = tfp.mcmc.NoUTurnSampler(
    toy_tf.target_log_prob_fn, step_size=1, # step size is the INITIAL one
    max_tree_depth=10, max_energy_diff=1000.0,
    unrolled_leapfrog_steps=1, parallel_iterations=10,
    experimental_shard_axis_names=None, name=None
)
@tf.function(experimental_compile=True)  # make it (much) faster -- https://github.com/tensorflow/probability/issues/728
def dualAveragingWithNUTS(delta):
    adaptive_nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=nuts_kernel_for_dual_averaging, num_adaptation_steps=n_tunning_eps, target_accept_prob=delta)
    return tfp.mcmc.sample_chain(
        num_results=n_tunning_eps,
        current_state=tf.zeros([n_chains, toy_tf.dim]),
        kernel=adaptive_nuts_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.step_size)
eps_opt_list = []
print("Tunning the step size epsilon...")
for delta in deltas:
    print("\tdelta = {}".format(delta))
    START = time.time()
    _, step_sizes = dualAveragingWithNUTS(delta)
    print('\t(tunning epsilon ok - eps = {} - took {:.4f}s)'.format(step_sizes[-1], time.time() - START))
    eps_opt_list.append(step_sizes[-1].numpy())
np.save('./simu_output/eps_opt_list_{}'.format(NOW),np.array(eps_opt_list))

# NUTS
@tf.function(experimental_compile=True)  # make it (much) faster -- https://github.com/tensorflow/probability/issues/728
def callNUTS(eps_opt):
    nuts_kernel = tfp.mcmc.NoUTurnSampler(
        toy_tf.target_log_prob_fn, step_size=eps_opt,
        max_tree_depth=10, max_energy_diff=1000.0,
        unrolled_leapfrog_steps=1, parallel_iterations=10,
        experimental_shard_axis_names=None, name=None
    )
    return tfp.mcmc.sample_chain(
        num_results=n_iter,
        current_state=tf.random.uniform((n_chains,toy_tf.dim)),  # differs from dualAveragingWithNUTS
        kernel=nuts_kernel,
        trace_fn=lambda _, pkr: [pkr.step_size,
                                 pkr.log_accept_ratio,
                                 pkr.leapfrogs_taken])
NUTS_out = [] # NOTE -- list of lists
grad_evals_NUTS = []
print("Running NUTS...")
for delta, eps_opt in zip(deltas, eps_opt_list):
    print("\tdelta = {}".format(delta))
    experiments = []
    START = time.time()
    samples, [step_size, log_accept_ratio, leapfrogs_taken] = callNUTS(eps_opt)
    print('\t{} chains ok - took {:.4f}s'.format(n_chains, time.time() - START))
    # from tensor of shape (n_iter,n_chains,dim) to list (of lists) with shape (n_chains,n_iter,dim):
    experiments = [list(x) for x in np.transpose(samples.numpy(), axes=[1,0,2])]
    NUTS_out.append(experiments)
    grad_evals_NUTS.append(list(tf.reduce_sum(leapfrogs_taken, axis=0).numpy())) # nb grad evals for calibrating metrics
np.save('./simu_output/NUTS_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(NUTS_out))

# eHMC
eHMC_out = [] # NOTE -- list of lists
grad_evals_eHMC = []
print("Running eHMC...")
for delta, eps_opt in zip(deltas, eps_opt_list):
    print("\tdelta = {}".format(delta))
    theta = np.random.rand(toy.dim)
    experiments = []
    grad_evals_inner = []
    for _ in range(n_chains):
        theta = np.random.rand(toy.dim)
        # compute the empirical longest batch distribution
        START = time.time()
        emp_batch_distr = computeEmpiricalBatchDistribution(theta=theta, eps=eps_opt, L=int(lambd/eps_opt),
                                                            K=n_emp_batch_distr, U=toy.U)
        print('\t\tempirical distr ok - took {:.4f}s'.format(time.time() - START))
        # eHMC
        START = time.time()
        samples, grad_count = eHMC(theta_0=theta, eps=eps_opt, emp_L=emp_batch_distr, N=n_iter,
                               U=toy.U, return_grad_count=True)
        print('\t\tchain ok - took {:.4f}s'.format(time.time() - START))
        experiments.append(samples)
        grad_evals_inner.append(grad_count) # nb grad evals for calibrating metrics
    eHMC_out.append(experiments)
    grad_evals_eHMC.append(grad_evals_inner)
np.save('./simu_output/eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(eHMC_out))

# compute metrics
##  NUTS
minESS_per_chain_NUTS, minESS_NUTS = [], []
ESJD_per_chain_NUTS, ESJD_NUTS = [], []
# TODO -- add KS distance
print("Computing metrics for NUTS...")
compute_for_each = True  # whether to compute the metrics for each individual chain (and delta) or only for each delta
for experiments,delta,grad_count_list in zip(NUTS_out,deltas,grad_evals_NUTS):  # one list of experiments per delta
    print("\tdelta = {}".format(delta))
    if compute_for_each:
        for sampling,grad_count in zip(experiments,grad_count_list):  # n_chains samplings per list of experiments
            X = np.array(sampling)  # X has shape (n_iter, dim)
            START = time.time()
            min_ess = minESS(X)/grad_count
            print('\t\tminESS for single chain = {}, took {:.4f}s'.format(min_ess, time.time()-START))
            minESS_per_chain_NUTS.append(min_ess)
            START = time.time()
            esjd = ESJD(X)/grad_count
            print('\t\tESJD for single chain = {}, took {:.4f}s'.format(esjd, time.time() - START))
            ESJD_per_chain_NUTS.append(esjd)
    X = np.array(experiments)
    START = time.time()
    min_ess = minESS(X)/sum(grad_count_list)
    print('\t\tcalibrated minESS = {}, took {:.4f}s'.format(min_ess, time.time() - START))
    minESS_NUTS.append(min_ess)
    START = time.time()
    esjd = ESJD(X)/sum(grad_count_list)
    print('\t\tcalibrated ESJD = {}, took {:.4f}s'.format(esjd, time.time() - START))
    ESJD_NUTS.append(esjd)
np.save('./simu_output/minESS_NUTS_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(minESS_NUTS))
np.save('./simu_output/minESS_PER_CHAIN_NUTS_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(minESS_per_chain_NUTS))
np.save('./simu_output/ESJD_NUTS_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(ESJD_NUTS))
np.save('./simu_output/ESJD_PER_CHAIN_NUTS_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(ESJD_per_chain_NUTS))
##  eHMC
minESS_per_chain_eHMC, minESS_eHMC = [], []
ESJD_per_chain_eHMC, ESJD_eHMC = [], []
# TODO -- add KS distance
print("Computing metrics for eHMC...")
compute_for_each = True  # whether to compute the metrics for each individual chain (and delta) or only for each delta
for experiments,delta,grad_count_list in zip(eHMC_out,deltas,grad_evals_eHMC):  # one list of experiments per delta
    print("\tdelta = {}".format(delta))
    if compute_for_each:
        for sampling,grad_count in zip(experiments,grad_count_list):  # n_chains samplings per list of experiments
            X = np.array(sampling)  # X has shape (n_iter, dim)
            START = time.time()
            min_ess = minESS(X)/grad_count
            print('\t\tcalibrated minESS for single chain = {}, took {:.4f}s'.format(min_ess, time.time()-START))
            minESS_per_chain_eHMC.append(min_ess)
            START = time.time()
            esjd = ESJD(X)/grad_count
            print('\t\tcalibrated ESJD for single chain = {}, took {:.4f}s'.format(esjd, time.time() - START))
            ESJD_per_chain_eHMC.append(esjd)
    X = np.array(experiments)
    START = time.time()
    min_ess = minESS(X)/sum(grad_count_list)
    print('\t\tminESS = {}, took {:.4f}s'.format(min_ess, time.time() - START))
    minESS_eHMC.append(min_ess)
    START = time.time()
    esjd = ESJD(X)/sum(grad_count_list)
    print('\t\tESJD = {}, took {:.4f}s'.format(esjd, time.time() - START))
    ESJD_eHMC.append(esjd)
np.save('./simu_output/minESS_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(minESS_eHMC))
np.save('./simu_output/minESS_PER_CHAIN_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(minESS_per_chain_eHMC))
np.save('./simu_output/ESJD_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(ESJD_eHMC))
np.save('./simu_output/ESJD_PER_CHAIN_eHMC_{}-chains_{}-iters_{}'.format(n_chains,n_iter,NOW),np.array(ESJD_per_chain_eHMC))

# visualization
x_for_each = np.repeat(deltas, n_chains)

plt.figure()
ax1 = plt.subplot(221)
if len(minESS_per_chain_eHMC) > 0:
    ax1.scatter(x_for_each, minESS_per_chain_eHMC, color='k')
ax1.plot(deltas, minESS_eHMC, 'rD-')
ax1.set_title('minESS, eHMC')

ax2 = plt.subplot(222, sharey=ax1)
if len(minESS_per_chain_NUTS) > 0:
    ax2.scatter(x_for_each, minESS_per_chain_NUTS, color='k')
ax2.plot(deltas, minESS_NUTS, 'rD-')
ax2.set_title('minESS, NUTS')

ax3 = plt.subplot(223, sharex=ax1)
if len(ESJD_per_chain_eHMC) > 0:
    ax3.scatter(x_for_each, ESJD_per_chain_eHMC, color='k')
ax3.plot(deltas, ESJD_eHMC, 'rD-')
ax3.set_xlabel('target acceptance rate')
ax3.set_title('ESJD, eHMC')

ax4 = plt.subplot(224, sharex=ax2, sharey=ax3)
if len(ESJD_per_chain_NUTS) > 0:
    ax4.scatter(x_for_each, ESJD_per_chain_NUTS, color='k')
ax4.plot(deltas, ESJD_NUTS, 'rD-')
ax4.set_xlabel('target acceptance rate')
ax4.set_title('ESJD, NUTS')

plt.show()