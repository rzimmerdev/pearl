import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('results.csv')

# Add idx within each (n_env, iteration) group
df['idx'] = df.groupby(['n_env', 'iteration']).cumcount()

# Filter to final step in each iteration
final = df[df['idx'] == 9]

# Compute average final wall time per n_env
wall_times = final.groupby('n_env')['wall_time'].mean().reset_index()

# Get baseline (wall time for n_env == 1)
baseline_time = wall_times.loc[wall_times['n_env'] == 1, 'wall_time'].values[0]

# Compute speedup
wall_times['speedup'] = baseline_time / wall_times['wall_time']

# Plot
plt.plot(wall_times['n_env'], wall_times['speedup'], marker='o')
plt.xlabel('n_env')
plt.ylabel('Speedup relative to n_env = 1')
plt.title('Speedup vs Number of Environments')
plt.grid(True)
plt.savefig('speedup.png')
