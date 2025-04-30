import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('results.csv')

# Compute idx (0 to 9) within each (n_env, iteration)
df['idx'] = df.groupby(['n_env', 'iteration']).cumcount()

# Average loss for each (n_env, idx)
grouped = df.groupby(['n_env', 'idx'])['loss'].mean().reset_index()

# Get log(loss) baseline at idx = 0 for each n_env
baseline = grouped[grouped['idx'] == 0][['n_env', 'loss']].rename(columns={'loss': 'baseline_loss'})
baseline['log_baseline'] = np.log(baseline['baseline_loss'])

# Merge and compute log difference
grouped = pd.merge(grouped, baseline[['n_env', 'log_baseline']], on='n_env')
grouped['log_relative_loss'] = np.log(grouped['loss']) - grouped['log_baseline']

# Scale idx by 10
grouped['x'] = grouped['idx'] * 10

# Plot
for n_env, sub_df in grouped.groupby('n_env'):
    plt.plot(sub_df['x'], sub_df['log_relative_loss'], label=f'n_env = {n_env}')

plt.xlabel('Episode')
plt.ylabel('Relative log(loss)')
plt.title('Log-relative Loss vs Episode by Number of Shared Envs')
plt.legend()
plt.grid(True)
plt.savefig('loss_episode.png')


