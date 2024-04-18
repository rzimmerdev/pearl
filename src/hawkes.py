import numpy as np
import matplotlib.pyplot as plt


class HawkesProcess:
    def __init__(self, alpha, beta, mu, T, d):
        self.alpha = alpha  # Intensity of self-exciting component
        self.beta = beta  # Decay parameter
        self.mu = mu  # Base intensity
        self.T = T  # Total time
        self.d = d  # Number of dimensions

    def intensity(self, t, events):
        intensity = self.mu
        for event_list in events:
            for event in event_list:
                intensity += self.alpha * np.exp(-self.beta * (t - event))
        return intensity

    def simulate(self):
        t = 0
        events = [[] for _ in range(self.d)]
        intensities = []

        while t < self.T:
            lambda_t = self.intensity(t, events)
            intensities.append(lambda_t)
            t += np.random.exponential(1 / lambda_t)
            if t < self.T:
                u = np.random.uniform()
                if u * lambda_t < self.mu:
                    dim = np.random.randint(self.d)
                    events[dim].append(t)

        return events, intensities


# Example usage
alpha = 0.5
beta = 1.0
mu = 0.1
T = 10.0
d = 3  # Number of dimensions

hawkes = HawkesProcess(alpha, beta, mu, T, d)
events, intensities = hawkes.simulate()

# Plot intensity function over time
t_values = np.linspace(0, T, 1000)
intensity_values = [hawkes.intensity(t, events) for t in t_values]

plt.figure(figsize=(10, 5))
for i in range(d):
    plt.scatter(events[i], [i] * len(events[i]), color='blue', label=f'Dimension {i}')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.title('Intensity Function of Hawkes Process')
plt.legend()
plt.grid(True)
plt.show()
