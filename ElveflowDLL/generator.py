import os
# Define the file name
output_file = "slanting_test_data.txt"
target_directory = r"C:\Users\arind_kv2r66v\projects\ML_GRIND\Stanford-Medicine-Research-Project\ElveflowDLL"

# Define pressure values up to 250
pressure_min = 0
pressure_max = 250
# Define time values for testing (fixed or varied as needed)

# Generate a sequence for control testing
sequence = [(i+1) for i in range(20)]

# Add cycles of increasing pressures with varying time durations


# Introduce random variations to pressure for real-world simulation
import random

random.seed(42)  # For reproducibility
arr = []
tot_time = 0
for i in range(50):
    q = random.choice(sequence)
    arr.append([random.uniform(pressure_min, pressure_max), q])
    tot_time+=q
    
file_path = os.path.join(target_directory, output_file)

# Write to a text file
with open(file_path, "w") as file:
    for pressure, time in arr:
        for i in range((5*time)):
            file.write(f"{float(pressure*i/(5*time))} {float(0.1)}\n")
        file.write(f"{0.0} {5.0}\n")
        tot_time+=5

print(tot_time)

print(f"Test data written to {target_directory}")
