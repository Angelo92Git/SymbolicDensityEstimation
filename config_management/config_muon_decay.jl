pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5

const CONFIG_sr = Dict(
    "binary_operators" => [+, -, *, /],
    "unary_operators" => [exp, log, pow2, pow3],
    "constraints" => [],
    "nested_constraints" => [exp => [exp => 0], log => [log => 0], pow2 => [pow2 => 0], pow3 => [pow3 => 0]],
    "maxsize" => 30,
    "ncycles_per_iteration" => 380,
    "parsimony" => 0.001,
    "warmup_maxsize_by" => 0.0,
    "adaptive_parsimony_scaling" => 1040,
    "parallelism_for_joint_sr" => :multithreading,
    "niterations_for_joint_sr" => 8000,
    "num_populations_for_joint_sr" => 15,
    "population_size_for_joint_sr" => 30,
    "batching" => true,
    "batch_size" => 128,
    "progress" => false,
    "verbosity" => false,
    "joint_use_frequency" => true,
    "joint_use_frequency_in_tournament" => true,
    "turbo" => false, # Enable turbo mode for faster computations
    "downsample_joint_data" => true,
    "num_samples_joint_data" => 10000, # Number of samples to use for joint data downsampling
    "elementwise_loss" => LossFunctions.L2DistLoss(),
)
