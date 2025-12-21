# python -m data_processing_scripts.gen_data_2d_gaussian_mixture

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def sample_two_modal_gaussian(n_samples=100000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Means of the two components
    means = [np.array([-4, 4]), np.array([4, -4])]

    # Covariance (0.1^2 * identity matrix)
    cov = np.array([[1, 0.8], [0.8, 1]])

    # Vectorized sampling
    # Determine number of samples for each component (approx equal prob)
    n_comp0 = np.random.binomial(n_samples, 0.5)
    n_comp1 = n_samples - n_comp0
    
    samples0 = np.random.multivariate_normal(means[0], cov, n_comp0)
    samples1 = np.random.multivariate_normal(means[1], cov, n_comp1)
    
    samples = np.vstack([samples0, samples1])
    np.random.shuffle(samples)
    
    return samples

def get_cluster_centroids(samples, labels):
    unique_labels = np.unique(labels)
    centroids = {}
    for lbl in unique_labels:
        if lbl == -1: continue
        centroids[lbl] = np.mean(samples[labels == lbl], axis=0)
    return centroids

def match_labels_to_reference(ref_centroids, current_centroids, tolerance=1.0):
    """
    Match current batch centroids to reference centroids.
    Returns a mapping from current_label -> ref_label.
    Raises ValueError if matching fails or is inconsistent.
    """
    mapping = {}
    
    # Check if number of clusters matches
    if len(ref_centroids) != len(current_centroids):
        raise ValueError(f"Inconsistent number of clusters: Reference has {len(ref_centroids)}, Batch has {len(current_centroids)}")
        
    used_ref_labels = set()
    
    for curr_lbl, curr_cent in current_centroids.items():
        # Find closest ref centroid
        best_ref_lbl = None
        min_dist = float('inf')
        
        for ref_lbl, ref_cent in ref_centroids.items():
            dist = np.linalg.norm(curr_cent - ref_cent)
            if dist < min_dist:
                min_dist = dist
                best_ref_lbl = ref_lbl
        
        if min_dist > tolerance:
             raise ValueError(f"Cluster centroid mismatch: Closest reference for cluster {curr_lbl} is dist {min_dist} > tol {tolerance}")
             
        if best_ref_lbl in used_ref_labels:
             raise ValueError(f"Ambiguous clustering: Multiple batch clusters map to reference cluster {best_ref_lbl}")
        
        mapping[curr_lbl] = best_ref_lbl
        used_ref_labels.add(best_ref_lbl)
        
    return mapping

def run_batched_dbscan(samples, eps, min_samples, n_batches=10):
    n_total = len(samples)
    batch_size = int(np.ceil(n_total / n_batches))
    
    all_labels = []
    ref_centroids = None
    
    print(f"Running Batched DBSCAN with {n_batches} batches...")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        
        if start_idx >= n_total:
            break
            
        batch_samples = samples[start_idx:end_idx]
        
        # Run DBSCAN on batch
        db = DBSCAN(eps=eps, min_samples=min_samples)
        batch_labels = db.fit_predict(batch_samples)
        
        # Calculate centroids
        batch_centroids = get_cluster_centroids(batch_samples, batch_labels)
        
        if i == 0:
            # First batch establishes reference
            ref_centroids = batch_centroids
            print(f"Batch 0 (Reference): Found {len(ref_centroids)} clusters.")
            current_batch_mapped_labels = batch_labels
        else:
            # Match to reference
            try:
                mapping = match_labels_to_reference(ref_centroids, batch_centroids, tolerance=2.0) # slightly loose tolerance
                
                # Apply mapping
                # Initialize with noise (-1)
                mapped_labels = np.full_like(batch_labels, -1)
                for old_lbl, new_lbl in mapping.items():
                    mapped_labels[batch_labels == old_lbl] = new_lbl
                
                # Keep original noise as noise (redundant but safe)
                mapped_labels[batch_labels == -1] = -1
                
                current_batch_mapped_labels = mapped_labels
                
            except ValueError as e:
                print(f"Batch {i} failed consistency check: {e}")
                raise e
        
        all_labels.append(current_batch_mapped_labels)
        
    return np.concatenate(all_labels)



# Generate and plot samples
samples = sample_two_modal_gaussian(seed=42)
samples = np.concatenate((samples, sample_two_modal_gaussian(seed=43)), axis=0)
x1 = samples[:, 0]
x2 = samples[:, 1]

# Save to CSV
df = pd.DataFrame({'x1': x1, 'x2': x2})
df.to_csv("./data/two_modal_samples.csv", index=False)


# --- assume `samples` is your (N×2) array from before ---

# 1. Run Batched DBSCAN
try:
    labels = run_batched_dbscan(samples, eps=5, min_samples=10, n_batches=10)
except ValueError as e:
    print(f"Detailed Error: {e}")
    exit(1)

# 2. Inspect how many points per cluster
unique, counts = np.unique(labels, return_counts=True)
print("cluster : count")
for lbl, cnt in zip(unique, counts):
    print(f"   {lbl: 2d}   : {cnt}")

# 3. (Optional) split samples by assigned label
# Only consider valid clusters (>= 0)
unique_valid = unique[unique >= 0]
clusters = {lbl: samples[labels == lbl] for lbl in unique_valid}

for cluster in clusters:
    # Cluster IDs are 0-indexed from DBSCAN usually, we map to 1-indexed for filename if desired, 
    # but the logic used cluster+1 previously.
    samples_cluster = np.array(clusters[cluster])
    x1 = samples_cluster[:, 0]
    x2 = samples_cluster[:, 1]
    df_cluster = pd.DataFrame({'x1': x1, 'x2': x2})
    df_cluster.to_csv(f"./data/two_modal_samples_cluster_{cluster+1}.csv", index=False)


import matplotlib.pyplot as plt
import numpy as np

# Assume 'samples' is your Nx2 array and 'labels' is a length-N array of cluster IDs
unique_labels = np.unique(labels)
cmap = plt.get_cmap("tab10")

# Plot each cluster separately with a label
for cluster_id in unique_labels:
    mask = labels == cluster_id
    plt.scatter(
        samples[mask, 0], samples[mask, 1],
        label=f"Cluster {cluster_id+1}",
        color=cmap(cluster_id % 10),
        alpha=0.6,
        s=20,               # Smaller marker size
        edgecolors='k',     # Black marker edge
        linewidths=0.5      # Optional: thinner edge
    )

plt.title("DBSCAN Clustering")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.legend(title="Cluster ID")
plt.show()



# plt.figure(figsize=(6, 6))
# plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
# plt.title("Samples from Two-Modal Gaussian Mixture")
# plt.xlabel("$x_1$")
# plt.ylabel("$x_2$")
# plt.grid(True)
# plt.axis("equal")
# plt.show()