import autograd.numpy as np
def log_accept_proba(theta, v, theta_new, v_new, U, M=None):
    """
    Compute log acceptance probability
    """
    if M is None:
        M = np.eye(len(theta))
    Minv = np.linalg.inv(M)
    v = v.reshape(-1,1)
    v_new = v_new.reshape(-1, 1)
    H_old = U(theta) + 0.5 * np.sum(v.T @ Minv @ v)
    H_new = U(theta_new) + 0.5 * np.sum(v_new.T @ Minv @ v_new)
    log_accept = min(0, H_old - H_new)
    return np.sum(log_accept)
    #using np.sum only to remove the matrix parentheses to obtain a real number
    # toy.U return a matrix of one element not a number

def H(theta, v, U, Minv):
    v = v.reshape(-1, 1)
    return np.sum(U(theta)) + 0.5 * np.sum(v.T @ Minv @ v)

if __name__=="__main__":
    # test code
    from toy import Simple2DGaussianMixture

    toy = Simple2DGaussianMixture()
    theta = np.array([1,2])
    v = np.array([3, 4])
    theta_new = np.array([5, 6])
    v_new = np.array([7, 8])

    print(log_accept_proba(theta, v, theta_new, v_new, toy.U))
