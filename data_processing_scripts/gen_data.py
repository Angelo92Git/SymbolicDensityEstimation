# To run in project root directory: python -m data_processing_scripts.gen_data
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy
import pandas as pd
from KDEpy import FFTKDE
from config_management.data_config_dijet import DataConfig
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

np.random.seed(42)

def kde1D(x, bandwidth, xbins, grid_tolerance, **kwargs):
    """Build 1D kernel density estimate (KDE)."""
    X = np.mgrid[x.min()-grid_tolerance:x.max()+grid_tolerance:xbins]
    kde_skl = FFTKDE(bw=bandwidth, **kwargs)
    kde_skl.fit(x.reshape(-1, 1))
    density = kde_skl.evaluate(X.reshape(-1, 1))
    return X, density

def read_file(file_path, columns):
    print("reading file...")
    df = pd.read_csv(file_path)
    samples = df[columns].to_numpy()
    return samples

def generate_joint(samples, save_prefix, kernel_type, bw_adj_joint, jxbins, grid_tolerance, filter, domain_extents=None, filter_threshold=None, domain_shrink_offset=None):
    print("generating joint distribution samples...")

    d = samples.shape[1]
    ####### for cls
    # slices = [slice(-12.0,4.0, jxbins), slice(-4.0, 12.0, jxbins)]
    # slices = [slice(-4.0,12.0, jxbins), slice(-12.0, 4.0, jxbins)]
    ### for cls_gm
    # slices = [slice(-12.0, 12.0, jxbins), slice(-12.0, 12.0, jxbins)]
    #######
    if domain_extents is not None:
        slices = [slice(domain_extents[i][0], domain_extents[i][1], jxbins) for i in range(d)]
    else:
        slices = [slice(np.min(samples[:, i])-grid_tolerance, np.max(samples[:, i])+grid_tolerance, jxbins) for i in range(d)]
    grids = np.mgrid[tuple(slices)]
    grid_coords = np.vstack([[grids[i].ravel()] for i in range(d)]).T
    bw = bw_adj_joint * scipy.stats.gaussian_kde(samples.T).scotts_factor()
    kde_all = FFTKDE(bw=bw, kernel=kernel_type)
    kde_all.fit(samples)
    zgrid = kde_all.evaluate(grid_coords)
    joint_data = np.concatenate((grid_coords, DataConfig.scaling_factor*zgrid.reshape(-1, 1)), axis=1)
    if filter and filter_threshold is not None:
        joint_data = joint_data[joint_data[:, -1] > filter_threshold]
    if DataConfig.domain_estimation:
        hull = ConvexHull(samples)
        hull_points = samples[hull.vertices]
        polygon = Polygon(hull_points)
        centroid = polygon.centroid
        shrunken_polygon = polygon.buffer(-domain_shrink_offset, join_style=1)
        points = joint_data[:, :joint_data.shape[1]-1]
        inside_mask = np.array([shrunken_polygon.contains(Point(p)) for p in points])
        joint_data = joint_data[inside_mask,:]
    np.savetxt(f"./data/processed_data/{save_prefix}_joint_data.csv", joint_data, delimiter=",")

def main():
    samples = read_file(DataConfig.data_file_path, DataConfig.columns)
    samples_min = np.min(samples, axis=0)
    samples_max = np.max(samples, axis=0)
    # Min-max scaling
    if DataConfig.min_max_scaling:
        print("Applying min-max scaling...")
        samples = (samples - samples_min) / (samples_max - samples_min)
    samples = DataConfig.scaling_factor * samples
    # Ensure the data is read correctly
    generate_joint(samples, save_prefix=DataConfig.processed_data_prefix, kernel_type=DataConfig.kernel_type, bw_adj_joint=DataConfig.bw_adj_joint, jxbins=DataConfig.jxbins, grid_tolerance=DataConfig.grid_tolerance, filter=DataConfig.filter, filter_threshold=DataConfig.filter_threshold, domain_shrink_offset=DataConfig.domain_shrink_offset)
    print("done!")

if __name__ == "__main__":
    main()
