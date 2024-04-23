import numpy as np
from plotly import graph_objects as go
from utils import mixtured_sampling, kl_divergence


def metropolis_hastings(p, q, qdraw, nsamp, xinit):
    # p(x), the distribution we want to sample from
    # q(x'|x), the proposal distribution
    # qdraw(x), a function to draw random samples from q(x'|x)
    # nsamp, the number of samples to generate
    # xinit, the initial state

    samples = np.empty(nsamp)
    x_prev = xinit
    accepted = 0

    for i in range(nsamp):
        x_star = qdraw(x_prev)
        p_star = p(x_star)
        p_prev = p(x_prev)
        a = p_star / p_prev
        if np.random.uniform() < a:
            samples[i] = x_star
            x_prev = x_star
            accepted += 1
        else:
            samples[i] = x_prev

    return samples, accepted / nsamp


rates = np.array([
    (0, 1),
    (4, 1),
])

target_samples = mixtured_sampling(rates, 1000)

# randomly initialize the proposal samples
proposal_samples = np.random.random(size=1000) * 10


def to_histogram(samples, bins):
    hist, _ = np.histogram(samples, bins=bins)
    return hist / np.sum(hist)


def sample_histogram(hist, size):
    return np.random.choice(np.arange(len(hist)), size=size, p=hist)


def pdf_histogram(hist, x):
    if x < 0 or x >= len(hist):
        return 0
    return hist[int(x)]


proposal_distribution = to_histogram(proposal_samples, np.arange(0, 10, 0.1))

steps = 1000

for i in range(steps):
    target_distribution = to_histogram(target_samples, np.arange(0, 10, 0.1))

    samples, acceptance_rate = metropolis_hastings(
        lambda x: pdf_histogram(target_distribution, x),
        lambda x: pdf_histogram(proposal_distribution, x),
        lambda x: np.random.choice(np.arange(len(proposal_distribution)), p=proposal_distribution),
        1000,
        np.random.choice(np.arange(len(proposal_distribution)), p=proposal_distribution),
    )

    if i % 10 == 0:
        print(acceptance_rate)
    if acceptance_rate >= 0.99:
        break

    proposal_samples = samples
    proposal_distribution = to_histogram(proposal_samples, np.arange(0, 10, 0.1))


# sample 1000 points from the proposal and test with the target
samples = sample_histogram(proposal_distribution, 1000)

print(
    kl_divergence(
        samples,
        target_samples,
    )
)
