pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5


const CONFIG_sr = Dict(
    "binary_operators" => [+, -, *, /],
    "unary_operators" => [exp, log, pow2, pow3],
    "constraints" => [],
    "nested_constraints" => [exp => [exp => 0], log => [log => 0], pow2 => [pow2 => 0], pow3 => [pow3 => 0]],
    "maxsize" => 50,
    "ncycles_per_iteration" => 380,
    "parsimony" => 0.001,
    "warmup_maxsize_by" => 0.0,
    "adaptive_parsimony_scaling" => 1040,
    "parallelism_for_marginal_sr" => :multithreading,
    "parallelism_for_conditional_sr" => :multithreading,
    "parallelism_for_joint_sr" => :multithreading,
    "niterations_for_marginal_sr" => 4000,
    "niterations_for_conditional_sr" => 4000,
    "niterations_for_joint_sr" => 8000,
    "num_populations_for_marginal_sr" => 15,
    "num_populations_for_conditional_sr" => 15,
    "num_populations_for_joint_sr" => 15,
    "population_size_for_marginal_sr" => 30,
    "population_size_for_conditional_sr" => 30,
    "population_size_for_joint_sr" => 30,
    "joint_expression_possibilities" => "cartesian", # "one_to_one" or "cartesian"    
    "joint_max_num_expressions_per_dim_and_slice" => 30, # use Inf to avoid limiting growth in number of expressions per dim and slice when multiplying conditionals and marginals
    "joint_max_num_expressions" => 450, # use Inf to avoid limiting number of expressions
    "batching" => true,
    "batch_size" => 128,
    "progress" => false,
    "verbosity" => false,
    "joint_use_frequency" => false,
    "joint_use_frequency_in_tournament" => false,
    "turbo" => false, # Enable turbo mode for faster computations
    "downsample_joint_data" => false,
    "num_samples_joint_data" => 1000, # Number of samples to use for joint data downsampling
    "elementwise_loss" => LossFunctions.L2DistLoss(),
    "scale_density" => true
)
