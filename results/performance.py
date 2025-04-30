import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('results.csv')

# Add idx within each (n_env, iteration) group
df['idx'] = df.groupby(['n_env', 'iteration']).cumcount()

# Filter to final loss in each iteration (idx == 9)
final_losses = df[df['idx'] == 9]

# Compute average final loss per n_env
avg_final_loss = final_losses.groupby('n_env')['loss'].mean().reset_index()

# Take log of the negative loss
avg_final_loss['log_neg_loss'] = -np.log(avg_final_loss['loss'])

# Plot
plt.plot(avg_final_loss['n_env'], avg_final_loss['log_neg_loss'], marker='o')
plt.axvline(8, color='blue', linestyle='dotted', label='physical cores on the machine')
plt.axvline(16, color='red', linestyle='dotted', label='total cores in the machine')

plt.xlabel('Number of shared environments')
plt.ylabel('-log(average final loss)')
plt.title('Performance per shared environment')
plt.legend()
plt.grid(True)
plt.savefig('performance.png')