import pandas as pd

# Read the CSV file
df = pd.read_csv('results.csv')

# Sort the dataframe by n_env and iteration to ensure the last rows are correctly selected
df = df.sort_values(by=['n_env', 'iteration'])

# Get the last row for each iteration within each n_env
last_iteration_rows = df.groupby(['n_env', 'iteration']).last().reset_index()

# Now group by n_env to get the mean and std dev of wall_time and loss
summary = last_iteration_rows.groupby('n_env').agg({
    'wall_time': ['mean', 'std'],
    'loss': ['mean', 'std']
}).reset_index()

# Flatten multi-level columns
summary.columns = ['n_env', 'wt_mean', 'wt_std', 'loss_mean', 'loss_std']

# Format output columns as mean ± std
summary['loss (mean ± std)'] = summary.apply(
    lambda row: f"{row['loss_mean']:.4f} ± {row['loss_std']:.4f}", axis=1)

summary['wall_time (mean ± std)'] = summary.apply(
    lambda row: f"{row['wt_mean']:.2f} ± {row['wt_std']:.2f}", axis=1)

# Optionally, you can drop the columns with mean and std values if you only want the formatted output
summary = summary[['n_env', 'loss (mean ± std)', 'wall_time (mean ± std)']]

# Create LaTeX table with custom design
latex_table = """
\\begin{table}[h!]
    \\centering
    \\begin{tabular}{|c|c|c|}
        \\hline
        \\textbf{n\_env} & \\textbf{loss (mean ± std)} & \\textbf{wall\_time (mean ± std)} \\\\
        \\hline
"""

# Add the rows of the DataFrame to the LaTeX table
for index, row in summary.iterrows():
    latex_table += f"        {row['n_env']} & {row['loss (mean ± std)']} & {row['wall_time (mean ± std)']} \\\\ \n"

# Close the table and add caption/label
latex_table += """
        \\hline
    \\end{tabular}
    \\caption{Summary of loss and wall time (mean ± std) for different n\_env values.}
    \\label{tab:summary}
\\end{table}
"""

# Print LaTeX table
print(latex_table)
