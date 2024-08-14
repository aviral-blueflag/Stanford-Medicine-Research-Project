import numpy as np
from scipy.linalg import solve_discrete_are

# Step 1: Load and parse the data
target_directory = r"C:\Users\arind_kv2r66v\projects\ML_GRIND\Stanford-Medicine-Research-Project\ElveflowDLL"

data = np.loadtxt(target_directory + '\conicle_output_08-09.txt', delimiter='\t')  # Assuming comma-separated
print(data)


time = data[:, 0]
set_pressure = data[:, 1]
actual_pressure = data[:, 2]
flow = data[:, 3]

# Step 2: Form state and input vectors
x = np.vstack((flow, actual_pressure)).T  # State vector: actual_pressure and flow
u = set_pressure  # Input vector


# Step 3: Estimate A and B matrices using linear regression
# x(t+1) = A*x(t) + B*u(t)
x_dot = np.diff(x, axis=0) / np.diff(time, axis=0)[:, None]  # Approximate derivative
print("x_dot shape", x_dot.shape)
# We need to align dimensions for linear regression
x = x[:-1]  # x(t)
u = u[:-1]  # u(t)
print("x shape", x.shape)
print("u shape", u.shape)
# Stack x and u for linear regression to find A and B
theta = np.linalg.lstsq(x, x_dot, rcond=None)[0] # Solve for A and B
print(theta)
A = theta
print("A", A.shape)

# Step 4: Define Q and R matrices
Q = np.eye(2)  # Penalize states equally
R = np.array([[1]])  # Penalize input

# Step 5: Solve the discrete-time LQR problem
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# Step 6: Apply the LQR control law
u_optimal = -K @ x.T

# The u_optimal is the optimal control input to apply for each state x
print("Optimal control input K:\n", K)
print("Optimal input u_optimal:\n", u_optimal)
