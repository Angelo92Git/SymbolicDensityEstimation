using CSV
using DataFrames
using Plots
gr()

# Define the function
f(x, y) = (10x^3 + 3x + y^2) / (x^2 + y^2 + x + y + 20)

# Generate 100 equally spaced points between 0 and 1
x_vals = range(0, 1, length=100)
y_vals = range(0, 1, length=100)

# Compute z for every (x, y) pair on the grid
data = [(x, y, f(x, y)) for x in x_vals, y in y_vals]

# Convert to a DataFrame
df = DataFrame(x = [d[1] for d in data[:]],
               y = [d[2] for d in data[:]],
               z = [d[3] for d in data[:]])

# reshape z values into a 100×100 matrix
Z = reshape(df.z, (100, 100))

surface(x_vals, y_vals, Z, xlabel="x", ylabel="y", zlabel="z", legend=false)

# Save to CSV
CSV.write("./data/test_data/grid_dataset.csv", df)

println("✅ Saved 1,000,000 points to grid_dataset.csv")
