
class DataConfig:
    data_file_path = "./data/Dijets.csv"
    processed_data_prefix = "dijet_minmax_scaled"
    columns = ['mjj', 'HT']
    jxbins = 10000j # Number of bins for joint distribution
    # (Not used)bw_adj_joint = 0.3 # Bandwidth adjustment for joint distribution KDE #120 # Old value (Not used)
    bw_adj_joint_range = (0.05, 0.65) # Range for bandwidth adjustment for joint distribution KDE
    cv_intervals = 11 # Number of intervals for cross-validation
    kernel_type = 'gaussian' # Kernel type for KDE
    reflection_lines = np.array([[7,0,0],[0,0,1]])
    grid_tolerance = 1e-10 # Tolerance for grid generation
    filter=False
    filter_threshold = None # Threshold for filtering out low probability values
    domain_estimation = False # Whether to estimate the domain of the data
    domain_shrink_offset = 1 # Offset to shrink the domain
    slices=None
    density_range_scaling_target = None # Target for density range scaling
    min_max_scaling = True