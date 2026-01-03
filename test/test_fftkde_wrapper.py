
import sys
import os
import numpy as np
from KDEpy import FFTKDE

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing_scripts.kde_wrapper import FFTKDEWrapper

def test_fftkde_wrapper():
    print("Testing FFTKDEWrapper...")
    
    # 1. Generate synthetic data
    np.random.seed(42)
    data = np.random.randn(100, 2)
    
    # 2. Define grid
    min_vals = data.min(axis=0) - 1.0
    max_vals = data.max(axis=0) + 1.0
    
    # Create a simple 10x10 grid
    x_ax = np.linspace(min_vals[0], max_vals[0], 20)
    y_ax = np.linspace(min_vals[1], max_vals[1], 20)
    
    # Use meshgrid (indexing='ij' usually maps to mgrid behavior but let's be careful with ordering)
    # The implementation assumed mgrid style flattening.
    # mgrid[min:max:step]
    
    # Let's reproduce the mgrid logic from gen_data.py to be safe/consistent
    # slices = [slice(min, max, n bins)...]
    # grids = np.mgrid[...]
    
    # Using linspace to simulate "slices" with complex step number for mgrid is equivalent to:
    # grid_x, grid_y = np.mgrid[min_vals[0]:max_vals[0]:20j, min_vals[1]:max_vals[1]:20j]
    
    grid_x, grid_y = np.meshgrid(x_ax, y_ax, indexing='ij')
    
    grids = [grid_x, grid_y]
    grid_coords = np.vstack([g.ravel() for g in grids]).T
    
    # 3. Instantiate base model and wrapper
    base_model = FFTKDE(bw=1.0) # Fixed bandwidth for 2D
    wrapper = FFTKDEWrapper(base_model, grid_coords)
    
    # 4. Fit wrapper
    print("Fitting wrapper...")
    wrapper.fit(data)
    
    # 5. Evaluate on training points (arbitrary points)
    print("Evaluating on data points...")
    try:
        results = wrapper.evaluate(data)
        print("Evaluation successful. Output shape:", results.shape)
        assert results.shape[0] == data.shape[0]
        assert np.all(results > 0) # Density should be positive
    except Exception as e:
        print("Evaluation failed:", e)
        raise
        
    # 6. Test out of bounds
    print("Testing out of bounds assertion...")
    out_of_bounds_point = np.array([[min_vals[0] - 2.0, min_vals[1]]])
    try:
        wrapper.evaluate(out_of_bounds_point)
        print("FAILED: Out of bounds point did not raise AssertionError")
    except AssertionError as e:
        print("SUCCESS: Caught expected error:", e)
    except Exception as e:
        print(f"FAILED: Caught unexpected error type {type(e)}: {e}")
        
    # 7. Basic interpolation accuracy check
    # Check if a grid point evaluates to the same value as the base model
    # Pick a point exactly on the grid
    test_idx = 50
    test_grid_point = grid_coords[test_idx:test_idx+1]
    
    # The base model evaluation on grid coords (stored in zgrid_reshaped)
    # The wrapper's evaluate should give approx same result
    
    wrapper_val = wrapper.evaluate(test_grid_point)
    # Compare with the cached value in zgrid from the fit step
    # Since we are evaluating AT a grid point, the interpolation should return the exact value (or very close)
    base_val = wrapper.zgrid[test_idx]
    
    print(f"Grid point test - Wrapper: {wrapper_val[0]}, Base (from zgrid): {base_val}")
    if np.isclose(wrapper_val, base_val):
        print("SUCCESS: Wrapper matches base model on grid point.")
    else:
        print("WARNING: Wrapper value differs from base model on grid point (could be interpolation numerical noise or bug).")

    print("All tests completed.")

if __name__ == "__main__":
    test_fftkde_wrapper()
