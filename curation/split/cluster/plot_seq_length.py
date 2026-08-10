import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
# Read CSV
df = pd.read_csv("/home/zwb/xlx/Interface_prediction/cluster/cluster_30_80_downloaded.csv")

# Compute the distribution of cluster_size
df['min_max_list'] = df['min_max'].apply(ast.literal_eval)
df['length_diff'] = df['min_max_list'].apply(lambda x:(x[1]-x[0])/(x[1]))
df['seq_length_list'] = df['seq_length'].apply(ast.literal_eval)

boxplot_data = df['seq_length_list'].tolist()
cluster_names = df['cluster_names'].tolist()

#print(len(cluster_names))
bins = [0,0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9, 1]
labels = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4","0.4-0.5", "0.5-0.6", "0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]

df['diff_bin'] = pd.cut(df['length_diff'], bins=bins, labels=labels, right=False)

#print(len(df['diff_bin']))
bin_counts = df['diff_bin'].value_counts().sort_index()
missing = df['diff_bin'].isna().sum()
#print("NaN count:", missing)
#print(bin_counts)
#print(top_clusters[1900:2000])
# Plot bar chart
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
    
plt.text(7,   
        25000,                 
        r"$Length Difference=\frac{Longest-Shortest}{Longest}$",                        # text of the annotation
        ha='center', va='bottom', fontsize=10)



xticks = np.arange(-0.5, len(bin_counts), 1)
xticklabels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # one extra tick on each side to avoid losing labels
plt.xticks(xticks, xticklabels, rotation=45)

plt.xlabel("Length Difference",fontdict=label_font)
plt.ylabel("Count of Clusters",fontdict=label_font)
plt.title("Length Difference Distribution",fontdict=title_font)
plt.tight_layout()
plt.savefig('cluster_30_80_downloaded_Diff_percentage Distribution.png')