using Statistics

pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5

# Loss Factory Function
function build_loss_function(total_volume::Float32)
    function custom_loss_closure(tree, dataset::Dataset, options)
        V_tot = total_volume

        preds, completed = eval_tree_array(tree, dataset.X, options)
        if !completed
            return Float32(1e9)
        end

        y = dataset.y
        is_sample = y .== -1.0f0

        # NLL on Samples
        sample_preds = preds[is_sample]
        if isempty(sample_preds)
            L_nll = 0.0f0
        else
            # Stable Log
            # Penalize <= 0 predictions heavily
            pos_mask = sample_preds .> 1e-9f0
            n_neg = count(.!pos_mask)
            L_penalty = n_neg * 1e6f0

            if any(pos_mask)
                L_nll = -mean(log.(sample_preds[pos_mask]))
            else
                L_nll = 0.0f0
            end
            L_nll += L_penalty
        end

        # Constraints on Grid
        grid_preds = preds[.!is_sample]
        if isempty(grid_preds)
            L_cons = 0.0f0
        else
            # Negativity (L2 on negative part)
            neg_vals = min.(grid_preds, 0.0f0)
            L_neg = sum(neg_vals .^ 2)

            # Integral Estimation (Monte Carlo over Batch)
            # Integral = Mean(f) * Volume_Domain
            ProbMass = mean(grid_preds) * V_tot
            L_norm = (ProbMass - 1.0f0)^2

            # Weighted Constraints
            # Reduced weights to allow exploration
            L_cons = 1.0f0 * L_neg + 1.0f0 * L_norm
        end

        return L_nll + L_cons
    end
    return custom_loss_closure
end

const CONFIG_sr = Dict(
    "binary_operators" => [+, -, *, /],
    "unary_operators" => [pow2, pow3, exp, log],
    "constraints" => [],
    "nested_constraints" =>
        [pow2 => [pow2 => 0], pow3 => [pow3 => 0], exp => [exp => 0], log => [log => 0]],
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
    "batch_size" => 1000,
    "progress" => false,
    "verbosity" => false,
    "joint_use_frequency" => true,
    "joint_use_frequency_in_tournament" => true,
    "turbo" => false, # Does not work here
    "downsample_joint_data" => true,
    "num_samples_joint_data" => 1000000,
    #"elementwise_loss" => hybrid_loss, # Replaced by loss_function factory
    "loss_factory" => build_loss_function,
    "use_pure_loss" => true, # Flag to trigger pure loss logic
    "use_hybrid_data" => true,
    "raw_samples_file" => "./data/Dijets.csv",
    "covariate_columns" => ["mjj", "HT"],
)
