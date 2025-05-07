import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv('results.csv')

# Group by n_env and iteration, aggregating wall_time and loss
grouped = df.groupby(['n_env', 'iteration']).agg({
    'wall_time': 'sum',
    'loss': 'sum'
}).reset_index()

# Calculate the loss per wall time (efficiency metric)
grouped['loss_per_wall_time'] = -np.log(grouped['loss']) / grouped['wall_time']
grouped['loss_per_wall_time'] /= grouped['loss_per_wall_time'][0]

# Now group by n_env to get the mean and std dev of both loss_per_wall_time and wall_time
summary = grouped.groupby('n_env').agg({
    'loss_per_wall_time': ['mean', 'std'],
    'wall_time': ['mean', 'std']
}).reset_index()

# Flatten multi-level columns
summary.columns = ['n_env', 'lpwt_mean', 'lpwt_std', 'wt_mean', 'wt_std']

# Format output columns as mean ± std
summary['loss_per_wall_time (mean ± std)'] = summary.apply(
    lambda row: f"{row['lpwt_mean']:.4f} ± {row['lpwt_std']:.4f}", axis=1)

summary['wall_time (mean ± std)'] = summary.apply(
    lambda row: f"{row['wt_mean']:.2f} ± {row['wt_std']:.2f}", axis=1)

# Select and print the final table
final_table = summary[['n_env', 'loss_per_wall_time (mean ± std)', 'wall_time (mean ± std)']]
print(final_table.to_string(index=False))
