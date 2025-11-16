import pandas as pd
import pprint

def main():

    df_paths = [
        '../logs/cluster_dijet_log_20251114_195733/joint_no_init/20251114_195759_m4lxU4/hall_of_fame.csv',
    ]

    post_fixes = [
        "",
    ]

    out_files = [
        "../data/pareto_results/dijet_results.txt",
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
