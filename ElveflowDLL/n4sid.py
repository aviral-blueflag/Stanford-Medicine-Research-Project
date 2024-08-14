import numpy as np
import pandas as pd
from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace

target_directory = r"C:\Users\arind_kv2r66v\projects\ML_GRIND\Stanford-Medicine-Research-Project\ElveflowDLL"

data = np.loadtxt(target_directory + '\conicle_output_08-09.txt', delimiter='\t')
print(data)

time = data[:, 0]
set_pressure = data[:, 1]
actual_pressure = data[:, 2]
flow = data[:, 3]

# Convert to Pandas DataFrame
y = pd.DataFrame(np.vstack((actual_pressure, flow)).T, columns=['actual_pressure', 'flow'])
u = pd.DataFrame(np.vstack((time, set_pressure)).T, columns=['time', 'set_pressure'])

# Check column names
print("u DataFrame columns:", u.columns)
print("y DataFrame columns:", y.columns)

# Define the order of the system you want to identify
system_order = 2  # Adjust this based on your system

# Create an instance of the NFourSID class
model = NFourSID(u, y, system_order)

# Perform system identification
A, B, C, D = model.system_matrices()

# Print the identified state-space matrices
print("A matrix:\n", A)
print("B matrix:\n", B)
print("C matrix:\n", C)
print("D matrix:\n", D)
