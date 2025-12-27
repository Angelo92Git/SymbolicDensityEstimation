import numpy as np
import sympy as sp

x, y = sp.symbols('x y')

# Generate data
y_low = -2
y_high = 2
x_low = -2
x_high = 2

f = (1/586.667)*(10 + 10*x**2 - 5*sp.cos(3*sp.pi*x-6.1) + 10*y**2 - 5*sp.cos(3*sp.pi*y-6.1))
marginal_x = sp.integrate(f, (y, y_low, y_high))
marginal_y = sp.integrate(f, (x, x_low, x_high))

f_func = sp.lambdify((x, y), f, 'numpy')
mx_func = sp.lambdify(x, marginal_x, 'numpy')
my_func = sp.lambdify(y, marginal_y, 'numpy')

x = np.linspace(x_low, x_high, 100)
y = np.linspace(y_low, y_high, 100)

# Generate joint distribution with probability density
# x_data_p = np.linspace(x_low, x_high, 10)
# y_data_p = np.linspace(y_low, y_high, 10)
# xg, yg = np.meshgrid(x_data_p, y_data_p)
# prob_val = f_func(xg, yg)
# prob_val = prob_val.flatten()
# joint_data_with_prob = np.vstack([xg.flatten(), yg.flatten(), prob_val]).T
# np.savetxt("./data/processed_data/modified_rastrigin_joint_data.csv", joint_data_with_prob, delimiter=",")

# Generate joint distribution for kde
x_data = np.linspace(x_low, x_high, 1000)
y_data = np.linspace(y_low, y_high, 1000)
prob_max = max(f_func(x_data, y_data))*1.05

def sample_pdf(n_samples=10000):
    samples = []
    while len(samples) < n_samples:
        x_sample = np.random.uniform(x_low, x_high)
        y_sample = np.random.uniform(y_low, y_high)
        z_sample = np.random.uniform(0, prob_max)
        if z_sample < f_func(x_sample, y_sample):
            samples.append([x_sample, y_sample])
    return np.array(samples)

samples = sample_pdf(1000000)
np.savetxt("./data/rastrigin.csv", samples, delimiter=",", header="x1,x2")
print(sp.latex(f.evalf(2)))