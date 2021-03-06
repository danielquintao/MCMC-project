import numpy as np

def minESS(X):
    """
    Effective Sample Size - Gelman et al, Bayesian Data Analysis, 15 Feb 2021
    params: X is tensor of shape (m,n,d) or (n,d)
        - m is the number of markov sequences (m=40 in the paper)
        - n is the length of each Markov sequence (n=20000 in the paper)
        - d is the dimention of target probability (e.g d=2)
    returns the min ESS across the d dimensions
    """
    m, n, d = X.shape if len(X.shape) == 3 else (1, X.shape[0], X.shape[1])
    if m == 1 and len(X.shape) == 3:
        X = X.squeeze(axis=0)  # we code the method to work for X of shape (n,d) if there is only one chain

    n_eff_list = []
    for k in range(d):
        # compute var_plus
        if m != 1:
            B = np.mean(X[:, :, k], axis=1)  # mean along columns, obtain shape(m,)
            B = n * np.var(B, ddof=1)
            W = np.var(X[:, :, k], axis=1, ddof=1)
            W = np.mean(W)
            var_plus = (n - 1) * W / n + B / n
        else:
            var_plus = np.var(X[:, k], ddof=1) # just variance

        #preparing to compute V_t, rho_t
        rho_t = 10
        sum_rho = 0
        last_rho_t = 0
        for t in range(1,n):
            if m != 1:
                V_t = np.sum((X[:,:n-t,k]-X[:,t:,k])**2, axis=1)
                V_t = np.mean(V_t)/(n-t)
            else:
                V_t = np.sum((X[:n - t, k] - X[t:, k]) ** 2)/(n-t)
            rho_t = 1 - V_t/2/var_plus
            sum_rho += rho_t
            if (last_rho_t + rho_t <= 0) and (t%2==1):
                break
            last_rho_t = rho_t
        n_eff_list.append(m*n/(1+2*sum_rho))

    return min(n_eff_list)



def ESJD(X):
    """
    Expected Square Jump Distance
    """
    m, n, d = X.shape if len(X.shape) == 3 else (1, X.shape[0], X.shape[1])
    if m == 1 and len(X.shape) == 3:
        X = X.squeeze(axis=0)  # we code the method to work for X of shape (n,d) if there is only one chain
    if m != 1:
        result = np.sum((X[:,:n-1,:]- X[:,1:,:])**2, axis=-1) # compute the square norm of the jumps
        result = np.mean(result, axis =-1)
        result = np.mean(result) # mean over m chains
    else:
        result = np.sum((X[:n - 1, :] - X[1:, :]) ** 2, axis=-1)  # compute the square norm of the jumps
        result = np.mean(result)
    return result


def KS(F, G):
    """
    Kolmogorov-Smirnov distance
    F, G are numpy array of shape (N,) representing two (possibly empirical) cumulative distributions
    """
    return (np.abs(F-G)).max()


if __name__ == "__main__":

    X = np.random.randn(4,20,2)
    print(minESS(X))
    print(ESJD(X))
    X = np.random.randn(20, 2)
    print(minESS(X))
    print(ESJD(X))