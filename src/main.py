import numpy as np
from plotly import graph_objects as go
from scipy.stats import norm


def get_random_genomes(n, size):
    return np.random.random(
        size=(n, size),
    )


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


def fittness(population, sample):
    hist_pred, _ = np.histogram(sample, bins=np.arange(min(min(sample), min(population)),
                                                       max(max(sample), max(population)) + 2))
    hist_true, _ = np.histogram(population, bins=np.arange(min(min(sample), min(population)),
                                                           max(max(sample), max(population)) + 2))

    return -kl_divergence(
        hist_true,
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


distributions = np.array([
    (0, 1),
    (4, 1),
])

true_distribution = mixtured_sampling(distributions, 1000)

n = 1000
size = 1000
selection_percentage = 0.1
selection = int(n * selection_percentage)
current_genomes = get_random_genomes(n, size) * 10

steps = 200
current_fittness = -np.inf
epsilon = 1e-9
gamma = 1
for i in range(steps):
    fittnesses = np.array([fittness(true_distribution, genome) for genome in current_genomes])

    # select 10 out of 100 with best fittness
    best_genomes = current_genomes[np.argsort(fittnesses)[-selection:]]

    if i % 10 == 0:
        print(fittnesses[np.argmax(fittnesses)])

    # break if no improvement
    if np.abs(fittnesses[np.argmax(fittnesses)] - current_fittness) < epsilon:
        break

    # crossover
    new_genomes = []

    for j in range(n - selection):
        parent1 = best_genomes[np.random.randint(selection)]
        parent2 = best_genomes[np.random.randint(selection)]
        child = np.zeros(size)
        for k in range(size):
            if np.random.random() < 0.5:
                child[k] = parent1[k]
            else:
                child[k] = parent2[k]
        new_genomes.append(child)

    # mutation
    for j in range(selection):
        if np.random.random() < 0.5:
            for k in range(size):
                new_genomes[j][k] += np.random.normal(0, gamma)

    new_genomes = np.concatenate((best_genomes, new_genomes))

    current_genomes = np.array(new_genomes)
    current_fittness = fittnesses[np.argmax(fittnesses)]

fittnesses = np.array([fittness(true_distribution, genome) for genome in current_genomes])
best_genomes = current_genomes[np.argsort(fittnesses)[-100:]]
best_genome = best_genomes[-1]

# print best genome
print("Best Fitness: ", np.max(fittnesses))

# plot both distributions
trace_true = go.Histogram(
    x=true_distribution,
    histnorm='probability',
    name='True Distribution',
    opacity=0.75,
    xbins=dict(
        start=min(true_distribution),
        end=max(true_distribution),
        size=0.1,
    ),
)

trace_pred = go.Histogram(
    x=best_genome,
    histnorm='probability',
    name='Predicted Distribution',
    opacity=0.75,
    xbins=dict(
        start=min(best_genome),
        end=max(best_genome),
        size=0.1,
    ),
)

layout = go.Layout(
    title='True vs Predicted Distribution',
    xaxis=dict(title='Value'),
    yaxis=dict(title='Probability'),
    barmode='overlay',
)

fig = go.Figure(data=[trace_true, trace_pred], layout=layout)
fig.show()
