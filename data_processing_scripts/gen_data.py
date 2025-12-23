# To run in project root directory: python -m data_processing_scripts.gen_data

import sys
## For Debugging:
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
##

import numpy as np
import scipy
import pandas as pd
import dill
from KDEpy import FFTKDE
from data_processing_scripts.kde_wrapper import FFTKDEWrapper, KDECVAdapter
from data_processing_scripts.cross_validation import cross_validate
import importlib
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

def create_grid(samples, jxbins, grid_tolerance, slices=None):
    d = samples.shape[1]
    if slices is None:
        slices = [slice(np.min(samples[:, i])-grid_tolerance, np.max(samples[:, i])+grid_tolerance, jxbins) for i in range(d)]
    grids = np.mgrid[tuple(slices)]
    grid_coords = np.vstack([[grids[i].ravel()] for i in range(d)]).T
    return grid_coords

def read_file(file_path, columns):
    print("reading file...")
    df = pd.read_csv(file_path)
    samples = df[columns].to_numpy()
    return samples

def generate_joint(samples, save_prefix, model_params, filter, filter_threshold=None, domain_estimation=False, domain_shrink_offset=None, density_range_scaling_target=None):
    print("generating joint distribution samples...")

    d = samples.shape[1]
    # This is specific to the FFTKDE implementation
    evaluation_grid = model_params['evaluation_grid']
    kernel_type = model_params['kernel_type']
    bw_adj_joint = model_params['bw_adj_joint']
    bw = bw_adj_joint * scipy.stats.gaussian_kde(samples.T).scotts_factor()
    print(f"Bandwidth: {bw}")
    kde_all = FFTKDE(bw=bw, kernel=kernel_type)
    kde_all.fit(samples)
    zgrid = kde_all.evaluate(evaluation_grid)
    
    # Wrap the model and verify
    wrapper = FFTKDEWrapper(kde_all, evaluation_grid).fit(samples)
    zgrid_wrapper = wrapper.evaluate(evaluation_grid)
    assert np.allclose(zgrid, zgrid_wrapper), "Wrapper evaluation does not match base model evaluation on grid points."
    print("Wrapper verification successful: zgrid matches zgrid_wrapper.")

    # Save models
    models_dir = "models"
    with open(f"{models_dir}/{save_prefix}_kde.pkl", "wb") as f:
        dill.dump(kde_all, f)
    with open(f"{models_dir}/{save_prefix}_kde_wrapped.pkl", "wb") as f:
        dill.dump(wrapper, f)
    print(f"Saved models to {models_dir}/{save_prefix}_kde.pkl and {models_dir}/{save_prefix}_kde_wrapped.pkl")

    joint_data = np.concatenate((evaluation_grid, zgrid.reshape(-1, 1)), axis=1)
    if filter and filter_threshold is not None:
        joint_data = joint_data[joint_data[:, -1] > filter_threshold]
    if domain_estimation:
        hull = ConvexHull(samples)
        hull_points = samples[hull.vertices]
        polygon = Polygon(hull_points)
        centroid = polygon.centroid
        shrunken_polygon = polygon.buffer(-domain_shrink_offset, join_style=1)
        points = joint_data[:, :joint_data.shape[1]-1]
        inside_mask = np.array([shrunken_polygon.contains(Point(p)) for p in points])
        joint_data = joint_data[inside_mask,:]
    if density_range_scaling_target is not None:
        max_density_value = np.max(joint_data[:, -1])
        min_density_value = np.min(joint_data[:, -1])
        density_range = max_density_value - min_density_value
        scale_factor = density_range_scaling_target / density_range
        print(f"Density range scaling factor: {scale_factor}")
        joint_data[:, -1] = joint_data[:, -1] * scale_factor
        np.savetxt(f"./data/processed_data/{save_prefix}_scale_factor.txt", [scale_factor])
    else:
        scale_factor = 1.0
        print(f"Density range scaling factor: {scale_factor}")
        np.savetxt(f"./data/processed_data/{save_prefix}_scale_factor.txt", [scale_factor])
    np.savetxt(f"./data/processed_data/{save_prefix}_joint_data.csv", joint_data, delimiter=",")

def load_models(save_prefix, models_dir="models"):
    """
    Loads the base KDE model and the wrapper from the specified directory.
    """
    with open(f"{models_dir}/{save_prefix}_kde.pkl", "rb") as f:
        kde_model = dill.load(f)
    with open(f"{models_dir}/{save_prefix}_kde_wrapped.pkl", "rb") as f:
        wrapper_model = dill.load(f)
    return kde_model, wrapper_model

def main(DataConfig):
    samples = read_file(DataConfig.data_file_path, DataConfig.columns)
    # 90-10 train-test split
    np.random.shuffle(samples)
    train_samples = samples[:int(len(samples)*0.9)]
    test_samples = samples[int(len(samples)*0.9):]
    np.savetxt(f"./data/processed_data/{DataConfig.processed_data_prefix}_test_samples.csv", test_samples, delimiter=",")

    # Avoid any data leakage by using the min and max of the training set only
    samples_min = np.min(train_samples, axis=0)
    samples_max = np.max(train_samples, axis=0)
    # Min-max scaling
    if DataConfig.min_max_scaling:
        print("Applying min-max scaling...")
        train_samples = (train_samples - samples_min) / (samples_max - samples_min)
        test_samples = (test_samples - samples_min) / (samples_max - samples_min)

    # Perform Cross-Validation to find best bw_adj_joint
    print("Performing Cross-Validation for Bandwidth Selection...")
    bw_adj_joint_range = DataConfig.bw_adj_joint_range
    cv_intervals = DataConfig.cv_intervals
    bw_multipliers = np.linspace(bw_adj_joint_range[0], bw_adj_joint_range[1], cv_intervals)
    
    # We pass the multipliers to the adapter, and the adapter multiplies by scott_factor internally
    cv_param_grid = {
        'bw_adj_joint': bw_multipliers,
        'kernel_type': [DataConfig.kernel_type]
    }
    
    # Generate grid for CV
    evaluation_grid = create_grid(samples, DataConfig.jxbins, DataConfig.grid_tolerance, slices=DataConfig.slices)
    
    kde_cv_config = {
        'evaluation_grid': evaluation_grid,
        'reflection_lines': DataConfig.reflection_lines
    }
    
    cv_model_factory = lambda params: KDECVAdapter(params, kde_cv_config)
    
    best_model, best_params, cv_results = cross_validate(train_samples, cv_model_factory, cv_param_grid, k=5)
    
    print(f"Cross-Validation Complete. Best Params: {best_params}")

    model_params = {
        'kernel_type': best_params['kernel_type'],
        'bw_adj_joint': best_params['bw_adj_joint'],
        'evaluation_grid': evaluation_grid,
        'reflection_lines': DataConfig.reflection_lines
    }
    
    generate_joint(train_samples, save_prefix=DataConfig.processed_data_prefix, model_params=model_params, filter=DataConfig.filter, filter_threshold=DataConfig.filter_threshold, domain_estimation=DataConfig.domain_estimation, domain_shrink_offset=DataConfig.domain_shrink_offset, density_range_scaling_target=DataConfig.density_range_scaling_target)
    
    # Read and print scale factor
    scale_factor = float(np.loadtxt(f"./data/processed_data/{DataConfig.processed_data_prefix}_scale_factor.txt"))
    print(f"Scale factor read from file: {scale_factor}")

    # Score model on test set
    print("Scoring model on test set...")
    _, wrapper = load_models(DataConfig.processed_data_prefix)
    densities = wrapper.evaluate(test_samples)
    
    # Handle zeros to avoid -inf
    densities = np.maximum(densities, 1e-10)
    log_likelihood = np.sum(np.log(densities))
    print(f"Test Set Log-Likelihood: {log_likelihood}")
    
    print("done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m data_processing_scripts.gen_data <config_module_name>")
        sys.exit(1)
        
    config_module_name = sys.argv[1]
    DataConfig = importlib.import_module(f"config_management.{config_module_name}").DataConfig
    main(DataConfig)
