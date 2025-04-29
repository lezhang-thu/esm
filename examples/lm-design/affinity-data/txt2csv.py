import pandas as pd

# Read the text file with whitespace as separator
df = pd.read_csv('SKEMPI_all_dg_avg.txt', sep=r'\s+', header=None, names=['A', 'B', 'Affinity'])

# Save to CSV
df.to_csv('SKEMPI.csv', index=False)
