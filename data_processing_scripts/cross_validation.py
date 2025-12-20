import numpy as np
import itertools
import copy

def get_param_combinations(param_grid):
    """
    Generates all combinations of parameters from a grid of parameters.
    param_grid: dict of {param_name: list_of_values}
    Returns: list of dicts [{param_name: value, ...}, ...]
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def k_fold_split(n_samples, k=5, seed=42):
    """
    Generates indices for K-fold cross-validation.
    Returns generator of (train_idx, test_idx).
    """
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        yield train_idx, test_idx
        current = stop

def cross_validate(data, model_factory, param_grid, k=5, seed=42):
    """
    Performs K-fold cross-validation to find the best hyperparameters.
    
    Args:
        data: (N, d) numpy array of training data.
        model_factory: function(params) -> model instance.
                       The model instance must have .fit(data) and .evaluate(data) methods.
                       .evaluate(data) should return PDF values (densities).
        param_grid: dict of {param_name: [list of values]}.
        k: number of folds.
        seed: random seed for reproducibility.
        
    Returns:
        best_model: Model instance trained on ALL data using best params.
        best_params: dict of best hyperparameters.
        results: dict containing details of the experiment.
    """
    param_combinations = get_param_combinations(param_grid)
    best_score = -np.inf
    best_params = None
    results = []
    
    print(f"Starting {k}-fold cross-validation with {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(k_fold_split(len(data), k, seed)):
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            # Create and train model
            model = model_factory(params)
            model.fit(train_data)
            
            # Evaluate (calculate mean log-likelihood on test fold)
            densities = model.evaluate(test_data)
            
            # Avoid log(0)
            densities = np.maximum(densities, 1e-15)
            log_likelihood = np.mean(np.log(densities))
            
            fold_scores.append(log_likelihood)
            
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"  -> Mean score: {mean_score:.4f} (std: {std_score:.4f})")
        
        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            
    print(f"\nBest score: {best_score:.4f} with params: {best_params}")
    
    # Retrain best model on full dataset
    best_model = model_factory(best_params)
    best_model.fit(data)
    
    return best_model, best_params, results
