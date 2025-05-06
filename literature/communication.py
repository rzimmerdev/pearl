import csv

import bibtexparser
import matplotlib.pyplot as plt
from collections import Counter

plt.style.use('./style.mplstyle')
BIBTEX_FILE = 'bibliography.bib'

with open(BIBTEX_FILE, 'r') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

centralization_counter = Counter()
synchronization_counter = Counter()

for entry in bib_database.entries:
    if 'communication' in entry:
        tags = [tag.strip().lower() for tag in entry['communication'].split(';') if tag.strip()]
        for tag in tags:
            if 'decentralized' in tag:
                centralization_counter['Decentralized'] += 1
            elif 'centralized' in tag:
                centralization_counter['Centralized'] += 1

    # Asynchronous vs Synchronous
    if 'communication' in entry:
        tags = [tag.strip().lower() for tag in entry['communication'].split(';') if tag.strip()]
        for tag in tags:
            if 'asynchronous' in tag:
                synchronization_counter['Asynchronous'] += 1
            elif 'synchronous' in tag:
                synchronization_counter['Synchronous'] += 1

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

labels_c, counts_c = zip(*centralization_counter.items())
axs[0].pie(counts_c, labels=labels_c, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
axs[0].set_title('Decentralized vs Centralized')

labels_s, counts_s = zip(*synchronization_counter.items())
axs[1].pie(counts_s, labels=labels_s, autopct='%1.1f%%', startangle=90, colors=['#99ff99','#ffcc99'])
axs[1].set_title('Asynchronous vs Synchronous')

plt.tight_layout()
plt.savefig("../paper/main/images/literature/communication.png")

csv_file = 'communication.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Label', 'Count'])

    for label, count in centralization_counter.items():
        writer.writerow([label, count])

    for label, count in synchronization_counter.items():
        writer.writerow([label, count])

print(f"CSV saved as {csv_file}")
