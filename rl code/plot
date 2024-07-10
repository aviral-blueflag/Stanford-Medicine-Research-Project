import matplotlib.pyplot as plt
import numpy as np

# Time array from 0 to 20 seconds
time = np.linspace(0, 20, 1000)

# Simulated pressure response (conceptual example)
# This is just a conceptual example; actual PID tuning may vary
desired_pressure = 5
pressure = 5 * (1 - np.exp(-0.3 * time))  # Simplified exponential approach to the setpoint

# Adding oscillation to simulate the PID adjustments
pressure += 0.5 * np.sin(2 * np.pi * 0.1 * time) * np.exp(-0.1 * time)

plt.figure(figsize=(10, 6))
plt.plot(time, pressure, label='Simulated Pressure Response')
plt.axhline(y=desired_pressure, color='r', linestyle='--', label='Desired Pressure')
plt.title('Pressure vs. Time with PID Control')
plt.xlabel('Time (seconds)')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)
plt.show()
