using Statistics

pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5

# Loss Factory Function
# Numerically safer / type-stable Loss Factory
function build_loss_function(total_volume::Real = 1.0)
    # Working precision (use Float64 to match dataset in your stack)
    V_tot = float(total_volume)          # domain volume as Float64
    const BIG_LOSS = 1.0e9               # returned when tree evaluation fails
    const EPS_LOG = 1.0e-12              # clamp for logs (prevents -Inf)
    const PENALTY_PER_NONPOS = 1.0e3     # penalty for non-positive sample preds
    const WEIGHT_NEG = 1.0               # weight for negativity penalty
    const WEIGHT_NORM = 1.0              # weight for normalization penalty
    const WEIGHT_NLL = 1.0               # weight for sample NLL

    function custom_loss_closure(tree, dataset::Dataset, options)
        # Evaluate tree
        preds_raw, completed = eval_tree_array(tree, dataset.X, options)
        if !completed
            return BIG_LOSS
        end

        # Ensure numeric type consistency
        preds = Float64.(preds_raw)    # predictions may come as Float64 already
        y = Float64.(dataset.y)        # dataset.y -> Float64

        # Identify samples vs grid points
        is_sample = y .==  -1.0

        ############
        # NLL on Samples (stable)
        ############
        L_nll = 0.0
        sample_preds = preds[is_sample]
        if !isempty(sample_preds)
            # mask of strictly positive predictions
            pos_mask = sample_preds .> 0.0
            n_nonpos = count(.!pos_mask)

            # penalty for non-positive predictions
            L_penalty = n_nonpos * PENALTY_PER_NONPOS

            if any(pos_mask)
                # clamp to EPS_LOG to avoid log(0); use safe_preds only for log
                safe_preds = max.(sample_preds[pos_mask], EPS_LOG)
                # negative log-likelihood (mean of -log p)
                L_nll = -mean(log.(safe_preds))
            else
                # no strictly positive predictions -> large NLL
                L_nll = PENALTY_PER_NONPOS * 10.0
            end

            L_nll += L_penalty
            L_nll *= WEIGHT_NLL
        end

        ############
        # Constraints on Grid
        ############
        L_cons = 0.0
        grid_preds = preds[.!is_sample]
        if !isempty(grid_preds)
            # negativity penalty (L2 on negative part)
            neg_vals = min.(grid_preds, 0.0)   # <= 0 values (negatives)
            L_neg = sum(neg_vals .^ 2)

            # Integral (Monte Carlo over the batch)
            ProbMass = mean(grid_preds) * V_tot
            L_norm = (ProbMass - 1.0)^2

            L_cons = WEIGHT_NEG * L_neg + WEIGHT_NORM * L_norm
        end

        return Float64(L_nll + L_cons)
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
