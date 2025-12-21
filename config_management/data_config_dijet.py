
class DataConfig:
    data_file_path = "./data/Dijets.csv"
    processed_data_prefix = "dijet_minmax_scaled"
    columns = ['mjj', 'HT']
    jxbins = 10000j # Number of bins for joint distribution
    bw_adj_joint_range = (0.2, 0.4)
    kernel_type = 'gaussian' # Kernel type for KDE
    slices=None
    density_range_scaling_target=None
    min_max_scaling = True