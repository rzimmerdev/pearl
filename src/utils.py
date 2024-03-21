import numpy as np
from scipy.stats import norm


def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)

    # Ensure histograms have the same shape
    assert p.shape == q.shape, "Histograms must have the same shape."

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10

    # Compute KL divergence
    kl_div = np.sum(p * np.log((p + epsilon) / (q + epsilon)))

    return kl_div


def fittness(target, pred):
    hist_pred, _ = np.histogram(pred, bins=np.arange(min(min(pred), min(target)),
                                                     max(max(pred), max(target)) + 2))
    hist_target, _ = np.histogram(target, bins=np.arange(min(min(pred), min(target)),
                                                         max(max(pred), max(target)) + 2))

    return -kl_divergence(
        hist_target,
        hist_pred,
    )


def mixtured_sampling(rates, size):
    """
    Mixture multiple normal distributions and sample from the convolved distribution.

    Parameters:
    - distributions: List of tuples, where each tuple contains the mean and standard deviation
                     for a normal distribution.
    - size: Number of points to sample from the convolved distribution.

    Returns:
    - samples: Array containing the sampled points from the mixtured distribution.
    """
    # Create a normal distribution for each rate
    distributions = [norm(loc=mean, scale=std) for mean, std in rates]

    # Sample from each distribution
    samples = [dist.rvs(size=size) for dist in distributions]

    # Mixture the samples
    mixtured_samples = np.concatenate(samples)

    # Sample size without reposition from the convolved distribution
    samples = np.random.choice(mixtured_samples, size=size, replace=False)

    return samples
