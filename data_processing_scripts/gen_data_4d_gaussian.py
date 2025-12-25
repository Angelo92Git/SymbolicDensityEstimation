# python -m data_processing_scripts.gen_data_4d_gaussian
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity
import sympy as sy
import scipy
from scipy.optimize import minimize
import pandas as pd
from pgmpy.estimators import PC
# import graphviz
import networkx as nx

np.random.seed(0)
mu = np.array([0,1,2,3])
cov = np.array([[3,1,0,0],[1,1,0,0],[0,0,4,2],[0,0,2,4]])
samples = np.random.multivariate_normal(mu, cov, size = 250000, check_valid='raise')
x1 = samples[:, 0]
x2 = samples[:, 1]
x3 = samples[:, 2]
x4 = samples[:, 3]

# Save to CSV
df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
df.to_csv("./data/4d_gaussian_samples.csv", index=False)

from pgmpy.estimators import PC
df = pd.DataFrame(samples[:250000,:], columns=['x1', 'x2', 'x3', 'x4'])
est = PC(data=df)
estimated_model = est.estimate(ci_test='pearsonr')

# Get a graphviz object.
# model_graph = estimated_model.to_graphviz()
# To open the plot
# model_graph.draw("./data/pgm.png", format='png', prog='dot')

# G_dag = estimated_model.to_dag()  # Might raise if edges can't be fully oriented

# plt.figure(figsize=(6,6))
# pos = nx.circular_layout(G_dag)

# nx.draw_networkx_nodes(G_dag, pos, node_color='lightgreen', node_size=1500)
# nx.draw_networkx_labels(G_dag, pos, font_size=12)
# nx.draw_networkx_edges(G_dag, pos, arrowstyle='->', arrowsize=20, edge_color='black')

# plt.title("Estimated DAG via NetworkX")
# plt.axis('off')
# plt.savefig('./data/pgm_dag.png', bbox_inches='tight')

df = pd.DataFrame({'x1': x1, 'x2': x2})
df.to_csv("./data/independent_set12_4d_gaussian_samples.csv", index=False)

df = pd.DataFrame({'x1': x3, 'x2': x4})
df.to_csv("./data/independent_set34_4d_gaussian_samples.csv", index=False)
print("Done")