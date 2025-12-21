using Pkg
Pkg.activate(".")
using SymbolicRegression
using CSV
using DataFrames
using Random
using StatsBase
using IterTools
using Base.Threads
using LoggingExtras
using Dates
using TimeZones
using FilePathsBase
using Serialization
using DynamicExpressions
using LoopVectorization
using LossFunctions

@assert length(ARGS) ≥ 1 "Missing required command-line argument: data_prefix"
@assert length(ARGS) ≥ 2 "Missing required command-line argument: config"
@assert length(ARGS) ≥ 3 "Missing required command-line argument: continue_flag"
@assert ARGS[3] == "true" || ARGS[3] == "false" "Expected true or false for third argument"

@assert isa(ARGS[1], String) "First argument must be a String"
prefix = ARGS[1]
if length(ARGS) ≥ 3
    continue_flag = ARGS[3] == "true"
end
if length(ARGS) == 4
    note = "_"*ARGS[4]
else
    note = ""
end

log_prefix = "cluster_"*prefix*note

include("./config_management/"*ARGS[2]*".jl")
cfg_sr=CONFIG_sr

data_path = "./data/processed_data"
files = readdir(data_path, join=true)

const cfg_data = Dict(
    "data_path_and_prefix" => data_path * "/" * prefix,
)

const cfg_log = Dict(
    "log_folder_prefix" => log_prefix,
    "continue_search" => continue_flag
)

if !cfg_log["continue_search"]
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    log_dir = "logs/" * log_prefix * "_log_$timestamp"
    if !isdir(log_dir)
        mkpath(log_dir)
    end
    state_init = nothing
else
    top_level_log_dir = "./logs"
    log_dir = joinpath(top_level_log_dir, maximum(filter(f -> startswith(f, log_prefix), readdir(top_level_log_dir))))
    load_path = joinpath(log_dir, "state.jls")
    state_init = deserialize(load_path)
    println("Successfully Loaded State!")
end

@info("log folder", log_dir=log_dir)

with_logger(FileLogger(joinpath(log_dir, "meta_data.log"))) do
    cfg_symbolized = Dict(Symbol(k) => v for (k, v) in cfg_sr)
    @info("Basic Information", thread_num=Threads.nthreads(), cfg_symbolized...)
end

# Create the DataFrame
df_j = CSV.read(cfg_data["data_path_and_prefix"] * "_joint_data.csv", DataFrame; header=false)

# Assume joint_data_matrix is a matrix with N columns,
# where the last column is the probability.
joint_data_matrix = Matrix(df_j)
n_cols = size(joint_data_matrix, 2)
x_dim = n_cols - 1
# Generate column names: [:x1, :x2, ..., :xD, :probability]
col_names = [Symbol("x$i") for i in 1:x_dim]
push!(col_names, :probability)
rename!(df_j, col_names)


if cfg_sr["downsample_joint_data"]
    # Set the random seed for reproducibility
    Random.seed!(123)
    # Number of rows to sample
    n = cfg_sr["num_samples_joint_data"]

    if n < nrow(df_j)
        # Sample rows without replacement
        df_j = df_j[shuffle(1:nrow(df_j))[1:n], :]
    else
        println("Did not need to downsample joint data, as the number of rows is less than the specified number of samples.")
    end
end

joint_data_x = Matrix(df_j[:, 1:(n_cols - 1)])
joint_data_y = df_j[:, n_cols]

CSV.write("./data/processed_data/"*prefix*"_joint_data_in_pipeline.csv", df_j)

joint_search_options = SymbolicRegression.Options(;
    binary_operators=cfg_sr["binary_operators"],
    unary_operators=cfg_sr["unary_operators"],
    populations=cfg_sr["num_populations_for_joint_sr"],
    population_size=cfg_sr["population_size_for_joint_sr"],
    constraints=cfg_sr["constraints"],
    nested_constraints=cfg_sr["nested_constraints"],
    output_directory=log_dir * "/joint_distribution_sr",
    maxsize=cfg_sr["maxsize"],
    ncycles_per_iteration=cfg_sr["ncycles_per_iteration"],
    parsimony=cfg_sr["parsimony"],
    warmup_maxsize_by=cfg_sr["warmup_maxsize_by"],
    adaptive_parsimony_scaling=cfg_sr["adaptive_parsimony_scaling"],
    progress=cfg_sr["progress"],
    verbosity=cfg_sr["verbosity"],
    use_frequency=cfg_sr["joint_use_frequency"],
    use_frequency_in_tournament=cfg_sr["joint_use_frequency_in_tournament"],
    batching=cfg_sr["batching"],
    batch_size=cfg_sr["batch_size"],
    turbo=cfg_sr["turbo"],
    elementwise_loss=cfg_sr["elementwise_loss"],
    return_state = true
)

println("Starting joint SR!")
now_utc = Dates.now()
et = TimeZone("America/New_York")
now_et = ZonedDateTime(now_utc, et)
println("Starting joint SR at $(now_utc) UTC")

search_result = equation_search(
    transpose(joint_data_x),
    joint_data_y;
    options=joint_search_options,
    parallelism=cfg_sr["parallelism_for_joint_sr"],
    niterations=cfg_sr["niterations_for_joint_sr"],
    saved_state = state_init
)

_ , joint_hall_of_fame = search_result
state = search_result

end_now_utc = Dates.now()
et = TimeZone("America/New_York")
end_now_et = ZonedDateTime(end_now_utc, et)
println("Completed joint SR at $(end_now_utc) UTC")
duration = end_now_utc - now_utc
println("Total duration: $(Dates.canonicalize(duration))")

# Save the joint hall of fame
save_path = joinpath(log_dir, "state.jls")
open(save_path, "w") do io
    serialize(io, state) # to load the data again use deserialze
end

save_path = joinpath(log_dir, "joint_hall_of_fame.jls")
open(save_path, "w") do io
    serialize(io, joint_hall_of_fame) # to load the data again use deserialze
end

println("Joint Hall of Fame saved to $(save_path)")
