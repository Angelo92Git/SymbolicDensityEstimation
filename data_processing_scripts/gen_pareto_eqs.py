# python -m data_processing_scripts.gen_pareto_eqs
import pandas as pd
import pprint
import os
import re
from pathlib import Path


def get_latest_hall_of_fame_path(prefix, logs_dir='logs', subfolder='joint_distribution_sr'):
    """
    Find the latest hall_of_fame.csv for a given log prefix.
    
    Directory structure expected:
    logs/{prefix}_log_{timestamp1}/{subfolder}/{timestamp2}_{suffix}/hall_of_fame.csv
    
    Args:
        prefix: Log prefix, e.g. "cluster_gaussian_mixture"
        logs_dir: Path to logs directory
        subfolder: Subfolder within the log directory (default: 'joint_distribution_sr')
    
    Returns:
        Path to the latest hall_of_fame.csv
    """
    logs_path = Path(logs_dir)
    
    # Pattern to match top-level directories: {prefix}_log_{timestamp}
    top_pattern = re.compile(rf'^{re.escape(prefix)}_log_(\d{{8}}_\d{{6}})$')
    
    # Find all matching top-level directories and sort by timestamp
    top_dirs = []
    for d in logs_path.iterdir():
        if d.is_dir():
            match = top_pattern.match(d.name)
            if match:
                top_dirs.append((match.group(1), d))
    
    if not top_dirs:
        raise FileNotFoundError(f"No log directories found matching prefix '{prefix}' in {logs_dir}")
    
    # Sort by timestamp (descending) and take the latest
    top_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_top_dir = top_dirs[0][1]
    
    # Now find the latest subdirectory within {subfolder}
    subfolder_path = latest_top_dir / subfolder
    if not subfolder_path.exists():
        raise FileNotFoundError(f"Subfolder '{subfolder}' not found in {latest_top_dir}")
    
    # Pattern to match subdirectories: {timestamp}_{suffix}
    sub_pattern = re.compile(r'^(\d{8}_\d{6})_\w+$')
    
    sub_dirs = []
    for d in subfolder_path.iterdir():
        if d.is_dir():
            match = sub_pattern.match(d.name)
            if match:
                sub_dirs.append((match.group(1), d))
    
    if not sub_dirs:
        raise FileNotFoundError(f"No timestamped subdirectories found in {subfolder_path}")
    
    # Sort by timestamp (descending) and take the latest
    sub_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_sub_dir = sub_dirs[0][1]
    
    hall_of_fame_path = latest_sub_dir / 'hall_of_fame.csv'
    if not hall_of_fame_path.exists():
        raise FileNotFoundError(f"hall_of_fame.csv not found in {latest_sub_dir}")
    
    return str(hall_of_fame_path)


def main():

    # Define prefixes for each dataset
    prefixes = [
        'jobresult_gaussian_mixture',
        'jobresult_dijet',
        'jobresult_gaussian_cluster_1',
        'jobresult_gaussian_cluster_2',
        'jobresult_gaussian_independent_set12',
        'jobresult_gaussian_independent_set34',
        'jobresult_gaussian_4d'
    ]

    post_fixes = [
        "",
        "",
        "",
        "",
        "",
        "",
        ""
    ]

    out_files = [
        "data/pareto_results/gaussian_mixture_results.py",
        "data/pareto_results/dijet_results.py",
        "data/pareto_results/gaussian_cluster_1_results.py",
        "data/pareto_results/gaussian_cluster_2_results.py",
        "data/pareto_results/gaussian_independent_set12_results.py",
        "data/pareto_results/gaussian_independent_set34_results.py",
        "data/pareto_results/gaussian_4d_results.py"
    ]

    # Validate array lengths
    assert len(post_fixes)==len(prefixes), "Bad match up between prefixes and postfixes"
    assert len(post_fixes)==len(out_files), "Bad match up between prefixes and out_files"


    for i in range(len(prefixes)):
        post_fix = post_fixes[i]
        
        try:
            df_path = get_latest_hall_of_fame_path(prefixes[i])
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping prefix '{prefixes[i]}'...")
            continue
            
        print(f"Loading: {df_path}")
        df = pd.read_csv(df_path)
        # Extract columns into Python lists
        raw_equations = df['Equation'].tolist()
        complexity = df['Complexity'].tolist()
        loss = df['Loss'].tolist()

        output_filename = out_files[i]
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        # Open the file in write mode
        with open(output_filename, 'w') as f:
            # raw_equations
            f.write('raw_equations' + post_fix + ' = [\n')
            for eq in raw_equations:
                f.write(f"    {eq!r},\n")
            f.write(']\n\n') # Added an extra newline for better separation

            # complexity
            f.write('complexity' + post_fix + ' = [\n')
            for c in complexity:
                f.write(f"    {c},\n")
            f.write(']\n\n')

            # loss
            f.write('loss' + post_fix + ' = [\n')
            for l in loss:
                f.write(f"    {l},\n")
            f.write(']\n\n')

        print(f"Data successfully written to {output_filename}")

if __name__ == '__main__':
    main()
