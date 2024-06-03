import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Paths to the JSON files
base_path = 'data/object_count_info/'
train_file_path = f'{base_path}train_object_counts.json'
val_seen_synonyms_path = f'{base_path}val_seen_synonyms_object_counts.json'
val_unseen_path = f'{base_path}val_unseen_object_counts.json'
val_seen_path = f'{base_path}val_seen_object_counts.json'
output_base_path = f'{base_path}cluster/'
# Load JSON data from the training file
with open(train_file_path, 'r') as f:
    train_data = json.load(f)

# Collect all unique labels from the training data
train_labels = list(train_data.keys())

# Load JSON data from the validation files
with open(val_seen_synonyms_path, 'r') as f:
    val_seen_synonyms = json.load(f)

with open(val_unseen_path, 'r') as f:
    val_unseen = json.load(f)

with open(val_seen_path, 'r') as f:
    val_seen = json.load(f)

# Collect all unique labels from the validation data
val_labels = set(val_seen_synonyms.keys()).union(val_unseen.keys()).union(val_seen.keys())
val_labels = list(val_labels)

# Load a pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode the training labels into embedding vectors
train_inputs = processor(text=train_labels, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    train_embeddings = model.get_text_features(**train_inputs)

# Encode the validation labels into embedding vectors
val_inputs = processor(text=val_labels, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    val_embeddings = model.get_text_features(**val_inputs)

# Convert embeddings to numpy arrays
train_embeddings = train_embeddings.cpu().numpy()
val_embeddings = val_embeddings.cpu().numpy()

# Determine the number of clusters for training data
num_clusters = 50

# Perform K-Means clustering on training data
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(train_embeddings)
train_labels_clusters = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Calculate the standard deviation of distances within each cluster
cluster_distances = {}
for i in range(num_clusters):
    cluster_distances[i] = np.max(np.linalg.norm(train_embeddings[train_labels_clusters == i] - cluster_centers[i], axis=1))

# Create a DataFrame with training labels and their cluster assignments
df_train_clusters = pd.DataFrame({'label': train_labels, 'cluster': train_labels_clusters})

# Assign validation labels to the nearest cluster or a new cluster based on distance threshold
threshold = 3.0  # Adjust this threshold as needed
val_labels_clusters = []
new_cluster_index = num_clusters

# Calculate distances from validation embeddings to cluster centers
closest_clusters, distances = pairwise_distances_argmin_min(val_embeddings, cluster_centers)

# Determine the clusters for validation labels
for i, dist in enumerate(distances):
    closest_cluster = closest_clusters[i]
    if dist > threshold and dist > cluster_distances[closest_cluster]:
        val_labels_clusters.append(new_cluster_index)
        new_cluster_index += 1
    else:
        val_labels_clusters.append(closest_cluster)
    
# Create a DataFrame with validation labels and their cluster assignments
df_val_clusters = pd.DataFrame({'label': val_labels, 'cluster': val_labels_clusters})

# Combine training and validation clusters into a single DataFrame
df_clusters = pd.concat([df_train_clusters, df_val_clusters], ignore_index=True)

# Save the clusters to a CSV file for readability
clusters_file = f'{output_base_path}label_clusters_{num_clusters}.csv'
df_clusters.to_csv(clusters_file, index=False)

# Output the cluster to object list information
cluster_info = {}
for label, cluster in zip(df_clusters['label'], df_clusters['cluster']):
    cluster = str(cluster)
    if cluster not in cluster_info:
        cluster_info[cluster] = []
    cluster_info[cluster].append(label)

# Save cluster info to a JSON file
cluster_info_file = f'{output_base_path}cluster_{num_clusters}_info.json'
with open(cluster_info_file, 'w') as f:
    json.dump(cluster_info, f, indent=4)

# Assign validation set to clusters
val_set_cluster_info = {}
for label, cluster in zip(val_labels, val_labels_clusters):
    cluster = str(cluster)
    if label not in val_set_cluster_info:
        val_set_cluster_info[label] = cluster_info[cluster]

# Save validation set cluster info to a JSON file
val_set_cluster_info_file = f'{output_base_path}val_set_cluster_{num_clusters}_info.json'
with open(val_set_cluster_info_file, 'w') as f:
    json.dump(val_set_cluster_info, f, indent=4)

print(f"Clustered labels have been saved to {clusters_file}")
print(f"Cluster information has been saved to {cluster_info_file}")
print(f"Validation set cluster information has been saved to {val_set_cluster_info_file}")
