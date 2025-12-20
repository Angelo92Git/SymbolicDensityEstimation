import pandas as pd
import pprint
import os

def main():

    df_paths = [
        '../logs/cluster_dijet_log_20251206_083931/joint_no_init/20251206_084008_WfYoSa/hall_of_fame.csv',
    ]

    post_fixes = [
        "",
    ]

    out_files = [
        "../data/pareto_results/dijet_results.py",
    ]

    # Load the Pareto front CSV
    assert len(post_fixes)==len(df_paths), "Bad match up between paths and postfixes"
    assert len(post_fixes)==len(out_files), "Bad match up between paths and postfixes"


    for i in range(len(post_fixes)):
        post_fix = post_fixes[i]
        df = pd.read_csv(df_paths[i])
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
