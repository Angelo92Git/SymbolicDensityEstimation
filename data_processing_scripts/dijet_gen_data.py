# python -m data_processing_scripts.dijet_gen_data.py

import numpy as np
from sklearn.neighbors import KernelDensity
import scipy
from scipy.stats import gaussian_kde
import pandas as pd
from tqdm import tqdm
from config_management.data_config_dijet import DataConfig
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib import cm  # For colormap
from KDEpy import FFTKDE
from scipy.interpolate import RegularGridInterpolator
from config_management.data_config_dijet import DataConfig

# Load dijets samples dataset

file_path = './data/Dijets.csv'
df = pd.read_csv(file_path)
print(df.head())
columns = ['mjj', 'HT']
samples = df[columns].to_numpy()

# scale mjj and HT between 0 and 1 from original samples dataset
mjj = df["mjj"].to_numpy()
HT = df["HT"].to_numpy()
mjj = (mjj - np.min(mjj)) / (np.max(mjj) - np.min(mjj))
HT = (HT - np.min(HT)) / (np.max(HT) - np.min(HT))

mjj = mjj.reshape(-1, 1)
HT = HT.reshape(-1, 1)

# Fit kernel density estimate on the scaled covariates
scaled_samples = np.hstack((mjj, HT))
d = scaled_samples.shape[1]
jxbins = DataConfig().jxbins
bw_adj_joint = DataConfig().bw_adj_joint
kernel_type = DataConfig().kernel_type
slices = [slice(-1.1, 1.1, jxbins), slice(-1.1, 1.1, jxbins)]
grids = np.mgrid[tuple(slices)]
grid_coords = np.vstack([[grids[i].ravel()] for i in range(d)]).T
bw = bw_adj_joint * scipy.stats.gaussian_kde(scaled_samples.T).scotts_factor()
kde_all = FFTKDE(bw=bw, kernel=kernel_type)
kde_all.fit(scaled_samples)
zgrid = kde_all.evaluate(grid_coords).reshape(grids[0].shape)
zgrid = np.clip(zgrid, 0, None)  # Ensure non-negative values

# Reflection trick demo on a hard boundary

# Assume xgrid, ygrid are your meshgrid arrays, zgrid is your function
m = 7

# Flatten the grid for easier computation
x_flat = grids[0].ravel()
y_flat = grids[1].ravel()

# Compute the reflection over y = m*x
d = (x_flat + m * y_flat) / (1 + m**2)
x_ref = 2 * d - x_flat
y_ref = 2 * d * m - y_flat

# Reshape to grid shape
x_ref_grid = x_ref.reshape(grids[0].shape)
y_ref_grid = y_ref.reshape(grids[1].shape)
z_ref_grid = zgrid  # or zgrid if you want signed values

# Corrected KDE using reflection trick

m1 = 7
m2 = 0

# 1. Reflect the original grid points
x_flat1 = grids[0].ravel()
y_flat1 = grids[1].ravel()
d1 = (x_flat1 + m1 * y_flat1) / (1 + m1**2)
x_ref1 = 2 * d1 - x_flat1
y_ref1 = 2 * d1 * m1 - y_flat1

x_flat2 = grids[0].ravel()
y_flat2 = grids[1].ravel()
d2 = (x_flat2 + m2 * y_flat2) / (1 + m2**2)
x_ref2 = 2 * d2 - x_flat2
y_ref2 = 2 * d2 * m2 - y_flat2

# 2. Interpolate the reflected function at the original grid points
# Prepare the interpolator for the original function
interp1 = RegularGridInterpolator(
    (grids[0][:,0], grids[1][0,:]),  # axes of the grid
    np.abs(zgrid),            # function values
    bounds_error=False,
    fill_value=0
)

interp2 = RegularGridInterpolator(
    (grids[0][:,0], grids[1][0,:]),  # axes of the grid
    np.abs(zgrid),            # function values
    bounds_error=False,
    fill_value=0
)


# For each original grid point, get the value of the reflected functions at that location
reflected_vals1 = interp1(np.vstack([x_ref1, y_ref1]).T)
reflected_vals_grid1 = reflected_vals1.reshape(grids[0].shape)

reflected_vals2 = interp2(np.vstack([x_ref2, y_ref2]).T)
reflected_vals_grid2 = reflected_vals2.reshape(grids[0].shape)

# 3. Add the reflected function to the original function
sum_grid = reflected_vals_grid1 + reflected_vals_grid2 + zgrid
mask = (grids[1] > 7 * grids[0]) | (grids[1] < 0)
sum_grid[mask] = 0

import numpy as np

# Flatten the grids and sum_grid
x_flat = grids[0].ravel()
y_flat = grids[1].ravel()
sum_flat = sum_grid.ravel()
mask_flat = mask.ravel()

# Select unmasked points
x_unmasked = x_flat[~mask_flat]
y_unmasked = y_flat[~mask_flat]
sum_unmasked = sum_flat[~mask_flat]

mask_x = x_unmasked > 0.6
mask_y = y_unmasked > 0.6
range_mask = mask_x | mask_y
x_final = x_unmasked[~range_mask]
y_final = y_unmasked[~range_mask]
sum_final = sum_unmasked[~range_mask]

# Stack into columns and save as CSV
data = np.column_stack((x_final, y_final, sum_final))
np.savetxt("./data/processed_data/dijet_joint_data.csv", data, delimiter=",", comments='')