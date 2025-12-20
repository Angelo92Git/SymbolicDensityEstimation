# python -m data_processing_scripts.gen_data_bi_modal_2D_gaussian_mixture.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def sample_two_modal_gaussian(n_samples=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    # Means of the two components
    means = [np.array([-4, 4]), np.array([4, -4])]

    # Covariance (0.1^2 * identity matrix)
    cov = np.array([[1, 0.8], [0.8, 1]])

    # Choose components: 0 or 1 with equal probability
    component_choices = np.random.choice([0, 1], size=n_samples)
    
    # Sample from the chosen components
    samples = np.array([
        np.random.multivariate_normal(mean=means[i], cov=cov)
        for i in component_choices
    ])
    
    return samples



# Generate and plot samples
samples = sample_two_modal_gaussian(n_samples=1000, seed=42)
x1 = samples[:, 0]
x2 = samples[:, 1]

# Save to CSV
df = pd.DataFrame({'x1': x1, 'x2': x2})
df.to_csv("./data/two_modal_samples.csv", index=False)


# --- assume `samples` is your (N×2) array from before ---

# 1. Instantiate & run DBSCAN in one go
db = DBSCAN(eps=5, min_samples=10)
labels = db.fit_predict(samples)
#   • labels[i] is the cluster ID for samples[i]
#   • noise points get label == -1

# 2. Inspect how many points per cluster
unique, counts = np.unique(labels, return_counts=True)
print("cluster : count")
for lbl, cnt in zip(unique, counts):
    print(f"   {lbl: 2d}   : {cnt}")

# 3. (Optional) split samples by assigned label
clusters = {lbl: samples[labels == lbl] for lbl in unique}

for cluster in clusters:
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