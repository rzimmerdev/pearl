import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('results.csv')

# Add idx within each (n_env, iteration) group
df['idx'] = df.groupby(['n_env', 'iteration']).cumcount()

# Filter to final step in each iteration
final = df[df['idx'] == 9]

# Compute average log loss and average wall_time per n_env
grouped = final.groupby('n_env').agg({
    'loss': lambda x: np.log(np.mean(x)),
    'wall_time': 'mean'
}).reset_index()

# Compute ratio
grouped['ratio'] = grouped['loss'] / grouped['wall_time']

# Divide by first value to get relative
grouped['ratio'] = -grouped['ratio'] / grouped['ratio'].iloc[0]

# Plot
plt.plot(grouped['n_env'], grouped['ratio'], marker='o')
plt.axhline(grouped['ratio'].max(), color='green', linestyle='dotted', label='max ratio')

plt.xlabel('n_env')
plt.ylabel('Average Negative log loss / Wall time')
plt.title('Efficiency of final loss per unit wall time')
plt.legend()
plt.grid(True)
plt.savefig('efficiency.png')
