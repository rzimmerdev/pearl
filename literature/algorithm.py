import csv

import bibtexparser
import matplotlib.pyplot as plt
from collections import Counter

# SETTINGS
BIBTEX_FILE = 'bibliography.bib'  # change to your actual file name
TAG_GROUP = 'algorithm'

# Read the BibTeX file
with open(BIBTEX_FILE, 'r') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

# Extract and count tags
counter = Counter()

for entry in bib_database.entries:
    if TAG_GROUP in entry:
        tags = entry[TAG_GROUP].split(';')
        tags = [tag.strip() for tag in tags if tag.strip()]
        counter.update(tags)

sorted_tags = list(reversed(counter.most_common()))
labels, counts = zip(*sorted_tags)

plt.figure(figsize=(8, 6))
plt.barh(labels, counts, color='skyblue')
plt.xlabel('Count')
plt.title(f'Distribution of {TAG_GROUP.capitalize()} Tags')
plt.tight_layout()
plt.savefig("../paper/main/images/literature/algorithm.png")

csv_file = 'algorithm.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Label', 'Count'])

    for label, count in counter.items():
        writer.writerow([label, count])


print(f"CSV saved as {csv_file}")

