import pickle
import pandas as pd
with open('/home/zwb/xlx/Interface_prediction/cluster/cluster_30_80_downloaded.pkl','rb') as d:
    clu_seq_dict=pickle.load(d)
#print(list(clu_seq_dict.items())[0:10])
cluster_names=list(clu_seq_dict.keys())
cluster_num=[]
seq_length=[]
seq_name=[]
min_max=[]
cluster_names.pop(0)
for cluster_name in cluster_names:
    cluster=clu_seq_dict[cluster_name]
    cluster_num.append(len(cluster))
    seq_length_in_cluster=[len(x)for x in list(cluster.values())]
    #print(seq_length_in_cluster)
    seq_length.append(seq_length_in_cluster)
    seq_name.append(list(cluster.keys()))
    min_max.append((min(seq_length_in_cluster),max(seq_length_in_cluster)))
#print(cluster_names)
#print(cluster_num)
#print(seq_length)
#print(seq_name)
#print(min_max)
print('total_seq',sum(cluster_num))
df = pd.DataFrame({
    "cluster_names": cluster_names,
    "cluster_size": cluster_num,
    "seq_name": seq_name,
    'seq_length':seq_length,
    'min_max':min_max
})
df = df.sort_values(by="cluster_size", ascending=False)
df.to_csv("cluster_30_80_downloaded.csv", index=True)