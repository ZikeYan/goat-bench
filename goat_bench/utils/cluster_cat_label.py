import json
import pandas as pd
from sklearn.cluster import KMeans
# from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel

base_path = 'data/object_count_info/cluster/'
# Load JSON data from the files
with open(f'{base_path}val_seen_synonyms_object_counts.json', 'r') as f:
    val_seen_synonyms = json.load(f)

with open(f'{base_path}val_unseen_object_counts.json', 'r') as f:
    val_unseen = json.load(f)

with open(f'{base_path}val_seen_object_counts.json', 'r') as f:
    val_seen = json.load(f)

# Collect all unique labels
all_labels = set(val_seen_synonyms.keys()).union(val_unseen.keys()).union(val_seen.keys())
all_labels = list(all_labels)

# # Load a pre-trained sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Encode the labels into embedding vectors
# embeddings = model.encode(all_labels)
# Load a pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode the labels into embedding vectors
inputs = processor(text=all_labels, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    embeddings = model.get_text_features(**inputs)

# Convert embeddings to numpy array
embeddings = embeddings.cpu().numpy()

# Determine the number of clusters
num_clusters = 50

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Create a DataFrame with labels and their cluster assignments
df_clusters = pd.DataFrame({'label': all_labels, 'cluster': labels})

# Save the clusters to a CSV file for readability
clusters_file = f'{base_path}label_clusters_{num_clusters}.csv'
df_clusters.to_csv(clusters_file, index=False)

# Output the cluster to object list information
cluster_info = {}
for label, cluster in zip(all_labels, labels):
    cluster = str(cluster)
    if cluster not in cluster_info:
        cluster_info[cluster] = []
    cluster_info[cluster].append(label)

# Print the cluster information
for cluster, items in cluster_info.items():
    print(f"Cluster {cluster}:")
    for item in items:
        print(f"  - {item}")

# Save cluster info to a JSON file
cluster_info_file = f'{base_path}cluster_{num_clusters}_info.json'
with open(cluster_info_file, 'w') as f:
    json.dump(cluster_info, f, indent=4)

print(f"Clustered labels have been saved to {clusters_file}")
print(f"Cluster information has been saved to {cluster_info_file}")
