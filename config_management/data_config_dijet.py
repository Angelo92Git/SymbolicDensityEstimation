
class DataConfig:
    data_file_path = "./data/Dijets.csv"
    processed_data_prefix = "dijet_minmax_scaled"
    columns = ['mjj', 'HT']
    jxbins = 10000j # Number of bins for joint distribution
    bw_adj_joint = 0.3 # Bandwidth adjustment for joint distribution KDE #120 # Old value
    kernel_type = 'gaussian' # Kernel type for KDE
