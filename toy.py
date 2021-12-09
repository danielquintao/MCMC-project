import autograd.numpy as np
from autograd import grad

class Simple2DGaussianMixture():
    """
    Example with mixture of two gaussians
    HOW TO USE:
    - initialize: toy = Simple2DGaussianMixture()
    - use it: toy.pi(x), toy.piUpToConstant(x), ...
    """

    def __init__(self):
        # centers
        self.mu1 = np.array([-0.5,-1])
        self.mu2 = np.array([1,1])
        # covariance matrices
        self.sigma1 = np.array([[1., 0.5],[0.5, 1.]])
        self.sigma1inv = np.linalg.inv(self.sigma1)
        self.sigma1det = np.linalg.det(self.sigma1)
        self.sigma2 = np.array([[.5, 0.1],[0.1, .5]])
        self.sigma2inv = np.linalg.inv(self.sigma2)
        self.sigma2det = np.linalg.det(self.sigma2)
        # weights for the gaussians
        self.w1 = 0.6
        self.w2 = 0.4

    def mvnpdf1(self, theta):
        theta = theta.flatten()
        vec = (theta-self.mu1).reshape(-1,1)
        return np.exp(-0.5*vec.T@self.sigma1inv@vec)/(2*np.pi*np.sqrt(self.sigma1det))

    def mvnpdf2(self, theta):
        theta = theta.flatten()
        vec = (theta - self.mu2).reshape(-1, 1)
        return np.exp(-0.5*vec.T@self.sigma2inv@vec)/(2*np.pi*np.sqrt(self.sigma2det))

    def pi(self, theta):
        # mixture of two gaussians
        return self.w1 * self.mvnpdf1(theta) + self.w2 * self.mvnpdf2(theta)

    def piUpToConstant(self, theta):
        return 5 * self.pi(theta)

    def U(self, theta):
        return -np.log(self.pi(theta))

    def UUpToConstant(self, theta):
        return -np.log(self.piUpToConstant(theta))

    def H(self, theta, v, M=np.eye(2)):
        v = v.reshape(-1, 1)
        return self.U(theta) + v.T @ M @ v / 2

    def HUpToConstant(self, theta, v, M=np.eye(2)):
        return self.UUpToConstant(theta) + v.T @ M @ v / 2

class MultivariateNormalDistribution():
    def __init__(self):
        self.dim = 10
        self.sigma = np.eye(self.dim)
        for i in range(self.dim):
            for j in range(i):
                self.sigma[i,j] = 0.99 ** abs(i-j)
                self.sigma[j,i] = 0.99 ** abs(i-j)
        self.sigmainv = np.linalg.inv(self.sigma)

    def U(self, theta):
        v = theta.reshape(-1,1)
        return v.T @ self.sigmainv @ v / 2

# TODO
class BayesianLogisticError():
    pass

class LectureExample():
    def __init__(self):
        mu1 = 2 * np.ones(2)
        cov1 = np.array([[1., 0.5],
                         [0.5, 1.]])
        mu2 = -mu1
        cov2 = np.array([[1., -0.1],
                         [-0.1, 1.]])

        mu3 = np.array([-1.5, 2.2])
        cov3 = 0.8 * np.eye(2)

        mu4 = np.array([2.5, -4.2])
        cov4 = 0.5 * np.eye(2)

        self.log_p = self.mixture(
            [self.multi_gauss(mu1, cov1), self.multi_gauss(mu2, cov2), self.multi_gauss(mu3, cov3), self.multi_gauss(mu4, cov4)],
            [0.25, 0.35, 0.3, 0.1])

    def multi_gauss(self, mu, sigma):
        """
        Inputs
        ----------
        mu: mean of the Gaussian distribution
        sigma: covariance matrix of the Gaussian distribution

        Outputs
        -------
        logp: opposite of the loglikelihood
        """

        def logp(x):
            k = mu.shape[0]
            cst = k * np.log(2 * np.pi)
            det = np.log(np.linalg.det(sigma))
            quad_term = np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), x - mu)
            return (cst + det + quad_term) * 0.5

        return logp

    def mixture(self, log_prob, weights):
        """
        Inputs
        ----------
        log_prob: opposite of the likelihood of each term
        weights: weights of the components of the mixture

        Outputs
        -------
        logp: opposite of the loglikelihood of the mixture
        """

        def logp(x):
            likelihood = 0
            for j in range(np.size(weights)):
                log_marginal = -log_prob[j](x)
                likelihood = likelihood + weights[j] * np.exp(log_marginal)

            return -np.log(likelihood)

        return logp

class LectureExample2(LectureExample):
    """
    The same example as Simple2DGaussianMixture, but in the course's notation
    """
    def __init__(self):
        super(LectureExample, self).__init__()
        mu1 = np.array([-0.5, -1.])
        mu2 = np.array([1., 1.])
        cov1 = np.array([[1., 0.5],[0.5, 1.]])
        cov2 = np.array([[.5, 0.1],[0.1, .5]])

        self.log_p = self.mixture(
            [self.multi_gauss(mu1, cov1), self.multi_gauss(mu2, cov2)],
            [0.6, 0.4])

