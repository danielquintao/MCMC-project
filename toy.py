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
        vec = (theta-self.mu1).reshape(-1,1)
        return np.exp(-0.5*vec.T@self.sigma1inv@vec)/(2*np.pi*np.sqrt(self.sigma1det))

    def mvnpdf2(self, theta):
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