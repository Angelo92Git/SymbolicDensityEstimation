pow2(x) = x^2
pow3(x) = x^3
pow4(x) = x^4
pow5(x) = x^5

function old_hybrid_loss(prediction, target, weight)
    # Target -1.0 indicates a raw sample point for NLL
    if target == -1.0
        # Robust NLL: Penalize <= 0 predictions heavily
        # This acts as a soft constraint for f(x) > 0
        if prediction <= 1e-9
            return 1e9 - 1e9 * prediction
        end
        return -log(prediction)
    else
        # Standard MSE for Grid points
        return (prediction - target)^2
    end
end

function hybrid_loss(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
      # Handle batching vs full dataset
      if idx === nothing
          X = dataset.X
          y = dataset.y
      else
          X = dataset.X[:, idx]
          y = dataset.y[idx]
      end
      
      prediction, flag = eval_tree_array(tree, X, options)
      if !flag
          return L(Inf)
      end
      is_neg_prediction = prediction .<= 0
      # Sentinel -1.0 logic for samples
      is_sample = y .< 0
      
      num_samples = sum(is_sample)
      num_grid = sum(.!is_sample)
      
      num_neg_predict_samples = sum(is_sample .&& is_neg_prediction)
      
      if num_samples > 0
          # NLL on samples (subset where y < 0)
          # Only take log of positive predictions
        #   valid_pos_preds = prediction[is_sample .&& .!is_neg_prediction]
        #   if !isempty(valid_pos_preds)
        #       nll_sum = sum(-log.(valid_pos_preds))
        #   else
        #       nll_sum = 0.0
        #   end
        #   nll_term = (nll_sum + num_neg_predict_samples * 1e9) / num_samples
          nll_term = (num_neg_predict_samples * 1e9) / num_samples
      else
          nll_term = 0.0
      end
      
      if num_grid > 0
          # MSE on grid (subset where y >= 0)
          mse_term = sum((prediction[.!is_sample] .- y[.!is_sample]) .^ 2) / num_grid
      else
          mse_term = 0.0
      end

      if num_samples > 0 && num_grid > 0
          return 0.1 * nll_term + mse_term
      elseif num_samples == 0 && num_grid > 0
          return mse_term
      elseif num_samples > 0 && num_grid == 0
          return 0.1 * nll_term
      else
        return L(Inf)
      end
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
    "niterations_for_joint_sr" => 16000,
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
    "num_samples_joint_data" => 1000000, # Number of samples to use for joint data downsampling
    # "elementwise_loss" => hybrid_loss,
    "loss_function" => hybrid_loss,
    "use_hybrid_data" => true,
    "raw_samples_file" => "./data/Dijets.csv",
    "covariate_columns" => ["mjj", "HT"],
)
