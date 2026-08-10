import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Read CSV
df = pd.read_csv("/home/zwb/xlx/Interface_prediction/cluster/cluster_30_80_downloaded.csv")

# Compute the distribution of cluster_size

bins = [0, 10,  100, 1000,10000,100000 ,1000000]
labels = ["1-10", "11-100", "101-1000", "1000-10000","10000-100000","100000+"]
df['size_bin'] = pd.cut(df['cluster_size'], bins=bins, labels=labels, right=True)

bin_counts = df['size_bin'].value_counts().sort_index()
# Draw the bar chart
plt.figure(figsize=(6,6))
plt.rcParams['font.family'] = 'Calibri'

title_font = {'color': 'black', 'weight': 'bold', 'size': 16}
label_font = {'color': 'black', 'weight': 'normal', 'size': 12}

bars=plt.bar(bin_counts.index.astype(str), bin_counts.values,width=1,color='#a493f9')

for bar, value in zip(bars, bin_counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,   # x coordinate: center of the bar
        bar.get_height(),                  # y coordinate: height of the bar
        str(value),                        # text of the annotation
        ha='center', va='bottom', fontsize=10
    )

plt.xlabel("Cluster Size (count of sequences)",fontdict=label_font)
plt.ylabel("Count of Clusters",fontdict=label_font)


xticks = np.arange(-0.5, len(bin_counts), 1)
xticklabels = [0,10,100,1000,10000,100000,1000000] # leave one extra slot on each side to avoid losing labels
plt.xticks(xticks, xticklabels, rotation=45)

plt.title("Distribution of Cluster Sizes",fontdict=title_font)
plt.tight_layout()
plt.savefig('cluster_30_80_cluster_downloaded_size.png')