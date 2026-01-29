gcl1_mse = 0.00016512596306781163
gcl2_mse = 0.00019224529635768678
gm_mse = 0.0004713606949003731
g12_mse = 0.00017640757718491757
g34_mse = 0.00014372948694124066
g4d_mse = 0.0010097743788591521

# f = g + h
# MSE_f = MSE_g + MSE_h (assuming errors in g and h are independent)
println("clustering MSE: $(gcl1_mse + gcl2_mse)")

# f = g * h
# MSE_f = 1/(4*\pi*\sqrt(1-0.8^2)) * (MSE_g + MSE_h) + MSE_g * MSE_h (assuming erros in g and h are independent)
println("structure learning MSE: $(1/(4*pi*sqrt(1-0.8^2)) * (g12_mse + g34_mse) + g12_mse * g34_mse)")
