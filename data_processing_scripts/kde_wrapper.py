import numpy as np
import scipy.interpolate
import scipy.stats
from KDEpy import FFTKDE

class FFTKDEWrapper:
    def __init__(self, base_model, grid_coords, reflection_lines=None):
        """
        Wrapper for FFTKDE to allow evaluation at arbitrary points via interpolation.
        Optionally applies reflection trick for boundary correction.
        
        Args:
            base_model: Instance of KDEpy.FFTKDE
            grid_coords: (N, d) array of grid points where base_model will be evaluated.
                         Must be an equidistant grid.
            reflection_lines: Optional (K, 3) array where each row is 
                              [slope, intercept, valid_above] defining a reflection line 
                              y = slope * x + intercept. valid_above=1 means the valid 
                              region is above the line, valid_above=0 means below.
                              Only supported for 2D grids. When provided, applies the
                              reflection trick to correct boundary bias.
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

        # Validate reflection_lines if provided
        if reflection_lines is not None:
            assert n_dims == 2, "Reflection trick only supported for 2D grids."
            reflection_lines = np.atleast_2d(reflection_lines)
            assert reflection_lines.shape[1] == 3, \
                "reflection_lines must have shape (K, 3) with columns [slope, intercept, valid_above]."

        # If all checks pass, initialize attributes
        self.base_model = base_model
        self.grid_coords = grid_coords
        self.reflection_lines = reflection_lines
        self.zgrid = None
        self.zgrid_corrected = None
        self.interpolator = None
        self.n_dims = n_dims
        self.grid_axes = grid_axes
        self.bounds = bounds
        self.grids = grids
            
    def fit(self, X, y=None):
        """
        Fits the base model and prepares the interpolator.
        Applies reflection trick if reflection_lines was provided.
        """
        # Fit the base model (which essentially just stores the data/computes stats)
        self.base_model.fit(X)
        
        # Evaluate on the fixed grid
        self.zgrid = self.base_model.evaluate(self.grid_coords)
        
        # Reshape zgrid for RegularGridInterpolator
        grid_shape = tuple(len(ax) for ax in self.grid_axes)
        self.zgrid_reshaped = self.zgrid.reshape(grid_shape)
        
        # Apply reflection trick if reflection_lines provided
        if self.reflection_lines is not None:
            final_grid = self._apply_reflection_trick()
        else:
            final_grid = self.zgrid_reshaped
        
        self.zgrid_corrected = final_grid
        
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            self.grid_axes, 
            final_grid, 
            bounds_error=True, 
            method='linear'
        )
        return self

    def _apply_reflection_trick(self):
        """
        Apply the reflection trick across all reflection lines.
        
        For each reflection line y = m*x + b, we:
        1. Reflect original grid points across the line
        2. Interpolate the KDE values at those reflected points
        3. Sum the reflected values with the original
        4. Zero out points outside the valid region
        """
        x_flat = self.grids[0].ravel()
        y_flat = self.grids[1].ravel()
        
        # Create interpolator for the original (uncorrected) KDE
        base_interp = scipy.interpolate.RegularGridInterpolator(
            (self.grid_axes[0], self.grid_axes[1]),
            np.abs(self.zgrid_reshaped),
            bounds_error=False,
            fill_value=0
        )
        
        # Start with the original KDE values
        sum_grid = self.zgrid_reshaped.copy()
        
        # Add reflected values for each reflection line
        for i in range(self.reflection_lines.shape[0]):
            m = self.reflection_lines[i, 0]  # slope
            b = self.reflection_lines[i, 1]  # intercept
            
            # Reflect points across line y = m*x + b
            # For a point (x, y), the reflection across line y = mx + b is:
            # d = (x + m*(y - b)) / (1 + m^2)
            # x_ref = 2*d - x
            # y_ref = 2*d*m + 2*b - y
            d = (x_flat + m * (y_flat - b)) / (1 + m**2)
            x_ref = 2 * d - x_flat
            y_ref = 2 * d * m + 2 * b - y_flat
            
            # Interpolate at reflected points
            reflected_vals = base_interp(np.vstack([x_ref, y_ref]).T)
            reflected_vals_grid = reflected_vals.reshape(self.grids[0].shape)
            
            sum_grid = sum_grid + reflected_vals_grid
        
        # Zero out points outside the valid region
        mask = self._compute_invalid_mask()
        sum_grid[mask] = 1e-50
        
        return sum_grid
    
    def _compute_invalid_mask(self):
        """
        Compute a mask for points outside the valid region defined by reflection lines.
        
        For each line, valid_above (third column) determines which side is valid:
        - valid_above=1: points above the line (y > mx + b) are valid
        - valid_above=0: points below the line (y < mx + b) are valid
        """
        mask = np.zeros(self.grids[0].shape, dtype=bool)
        
        for i in range(self.reflection_lines.shape[0]):
            m = self.reflection_lines[i, 0]
            b = self.reflection_lines[i, 1]
            valid_above = self.reflection_lines[i, 2]
            
            if valid_above:
                # Valid region is above the line, so below is invalid
                line_mask = self.grids[1] < m * self.grids[0] + b
            else:
                # Valid region is below the line, so above is invalid
                line_mask = self.grids[1] > m * self.grids[0] + b
            mask = mask | line_mask
            
        return mask

    def evaluate(self, X):
        """
        Evaluates the model at arbitrary points X using interpolation.
        
        Args:
            X: (M, d) array of points.
        """
        for i in range(X.shape[0]):
            for d in range(self.n_dims):
                if not (self.bounds[d][0] <= X[i, d] <= self.bounds[d][1]):
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
        
        # Get grid and optional reflection_lines from config
        evaluation_grid = self.config['evaluation_grid']
        reflection_lines = self.config.get('reflection_lines', None)
        
        base_model = FFTKDE(bw=bw, kernel=kernel_type).fit(X)
        self.model = FFTKDEWrapper(base_model, evaluation_grid, reflection_lines=reflection_lines).fit(X)
        return self

    def evaluate(self, X):
        return self.model.evaluate(X)
