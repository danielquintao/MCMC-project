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
n_iter_ground_truth = 2000
n_burnin_ground_truth = 2000

toy = MultivariateNormalDistribution()
toy_tf = MultivariateNormalDistributionTensorflowVersion()

# recover saved data
eps_opt_list = np.load('./simu_output/eps_opt_list_{}.npy'.format(NOW))
NUTS_out_arr = np.load('./simu_output/NUTS_{}-chains_{}-iters_{}.npy'.format(n_chains,n_iter,NOW))
eHMC_out_arr = np.load('./simu_output/eHMC_{}-chains_{}-iters_{}.npy'.format(n_chains,n_iter,NOW))

# compute a reference sampling
@tf.function(experimental_compile=True)  # make it (much) faster -- https://github.com/tensorflow/probability/issues/728
def computeReference():
    nuts_kernel = tfp.mcmc.NoUTurnSampler(
        toy_tf.target_log_prob_fn, step_size=1,
        max_tree_depth=10, max_energy_diff=1000.0,
        unrolled_leapfrog_steps=1, parallel_iterations=10,
        experimental_shard_axis_names=None, name=None
    )
    adaptive_nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=nuts_kernel, num_adaptation_steps=n_burnin_ground_truth, target_accept_prob=0.65)
    return tfp.mcmc.sample_chain(
        num_results=n_iter_ground_truth,
        current_state=tf.random.uniform((1,toy_tf.dim)),  # differs from dualAveragingWithNUTS
        num_burnin_steps=n_burnin_ground_truth,
        kernel=adaptive_nuts_kernel,
        trace_fn=lambda _, pkr: [pkr.inner_results.log_accept_ratio]
    )
START = time.time()
print("computing ground truth for KS...")
reference, _ = computeReference()
print("ok ({}s)".format(time.time()-START))
reference = reference.numpy().squeeze()  # reference becomes a np array of shape (n_iter_ground_truth, dim)
print(reference.shape)

cdfs_ref = []  # shape (dim, bins)
for d in range(toy_tf.dim):
    hist, _ = np.histogram(reference[:,d], bins)
    hist = hist/sum(hist)
    cdfs_ref.append(np.cumsum(hist))

np.save('./simu_output/Groundtruth_KS_{}_iters'.format(n_iter_ground_truth), cdfs_ref)
plt.figure()
plt.plot(bins[:-1], cdfs_ref[0])
plt.show()


