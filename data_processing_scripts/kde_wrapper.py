import numpy as np
import scipy.interpolate
import scipy.stats
from KDEpy import FFTKDE

class FFTKDEWrapper:
    def __init__(self, base_model, grid_coords):
        """
        Wrapper for FFTKDE to allow evaluation at arbitrary points via interpolation.
        
        Args:
            base_model: Instance of KDEpy.FFTKDE
            grid_coords: (N, d) array of grid points where base_model will be evaluated.
                         Must be an equidistant grid.
        """
        # Validate grid properties immediately (Fail fast)
        n_dims = grid_coords.shape[1]
        grid_axes = []
        bounds = []
        
        # 1. Infer axes and check equidistance
        for d in range(n_dims):
            unique_vals = np.unique(grid_coords[:, d])
            
            # Assert equidistance (constant step size)
            if len(unique_vals) > 1:
                diffs = np.diff(unique_vals)
                # Check if all differences are close to the mean difference
                assert np.allclose(diffs, diffs[0]), f"Grid axis {d} is not equidistant."
                
            grid_axes.append(unique_vals)
            bounds.append((unique_vals.min(), unique_vals.max()))

        # 2. Check total number of points matches product of axis lengths
        expected_n_points = np.prod([len(ax) for ax in grid_axes])
        assert grid_coords.shape[0] == expected_n_points, \
            f"grid_coords length {grid_coords.shape[0]} does not match product of unique values per dimension {expected_n_points}"
        
        # 3. Assert grid_coords are sorted in C-style order (last index fastest)
        # This is crucial for RegularGridInterpolator which expects data on the meshgrid(indexing='ij') flattened
        grids = np.meshgrid(*grid_axes, indexing='ij')
        expected_grid_coords = np.vstack([g.ravel() for g in grids]).T
        
        assert np.allclose(grid_coords, expected_grid_coords), \
            "grid_coords must be sorted in C-style lexicographical order (consistent with np.mgrid and meshgrid indexing='ij')."

        # If all checks pass, initialize attributes
        self.base_model = base_model
        self.grid_coords = grid_coords
        self.zgrid = None
        self.interpolator = None
        self.n_dims = n_dims
        self.grid_axes = grid_axes
        self.bounds = bounds
            
    def fit(self, X, y=None):
        """
        Fits the base model and prepares the interpolator.
        """
        # Fit the base model (which essentially just stores the data/computes stats)
        self.base_model.fit(X)
        
        # Evaluate on the fixed grid
        self.zgrid = self.base_model.evaluate(self.grid_coords)
        
        # Reshape zgrid for RegularGridInterpolator
        # The shape should match the lengths of grid_axes
        grid_shape = tuple(len(ax) for ax in self.grid_axes)
        
        # We need to make sure the zgrid is reshaped correctly corresponding to grid_axes order.
        # grid_coords usually comes from np.mgrid, which produces a specific order.
        # If grid_coords is (N, d) where N is product of dims.
        # We generated grids in gen_data.py using:
        # grids = np.mgrid[...]
        # grid_coords = np.vstack([[grids[i].ravel()] for i in range(d)]).T
        # This implies standard C-order flattening (default for ravel/reshape).
        
        # However, np.mgrid output: grids[0] varies slowest (axis 0), grids[1] varies faster.
        # Wait, np.mgrid[0:2, 0:2] -> 
        # grids[0]: [[0, 0], [1, 1]]
        # grids[1]: [[0, 1], [0, 1]]
        # ravel: 0 0 1 1 (axis 0), 0 1 0 1 (axis 1)
        # So evaluating on this gives a vector corresponding to this order.
        # Refilling it into shape (2, 2) naturally matches.
        
        self.zgrid_reshaped = self.zgrid.reshape(grid_shape)
        
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            self.grid_axes, 
            self.zgrid_reshaped, 
            bounds_error=True, 
            method='linear'
        )
        return self

    def evaluate(self, X):
        """
        Evaluates the model at arbitrary points X using interpolation.
        
        Args:
            X: (M, d) array of points.
        """
        # Check bounds
        for i in range(X.shape[0]):
            for d in range(self.n_dims):
                if not (self.bounds[d][0] <= X[i, d] <= self.bounds[d][1]):
                    # Check tolerance? Ideally strict as requested "assert that any arbitray evaluation points do not lie outside this grid"
                     raise AssertionError(f"Point {X[i]} is outside grid bounds in dimension {d}. Range: {self.bounds[d]}")
                     
        return self.interpolator(X)

class KDECVAdapter:
    def __init__(self, hyperparams, config):
        self.hyperparams = hyperparams
        self.config = config
        self.model = None

    def fit(self, X):
        kernel_type = self.hyperparams['kernel_type']
        bw_adj_joint = self.hyperparams['bw_adj_joint']
        
        # Calculate bandwidth
        bw = bw_adj_joint * scipy.stats.gaussian_kde(X.T).scotts_factor()
        
        # Get grid from config
        grid_coords = self.config['grid_coords']
        
        base_model = FFTKDE(bw=bw, kernel=kernel_type).fit(X)
        self.model = FFTKDEWrapper(base_model, grid_coords).fit(X)
        return self

    def evaluate(self, X):
        return self.model.evaluate(X)
