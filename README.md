# Python implementation of *Faster Hamiltonian Monte Carlo by Learning Leapfrog Scale¹*
### ¹ *(C Robert, C Wu, J Stoehr, CP Robert - 2019)*


We implemented and tested the Empirical Hamiltonian Monte Carlo (eHMC) and Partially Refreshed Hamiltonian Monte Carlo
(prHMC) proposed in the paper mentioned above. These are HMC-based  Markov Chain Monte Carlo (MCMC) methods that allow 
us to sample from complicated distributions (which we may know only up to a constant) efficiently.

We ran a simpler version of the Experiment 1 of the paper, using
```tensorflow_probability``` implementtion of NUTS as benchmark.

![img.png](img.png)