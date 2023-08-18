import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the cubic function
def cubic_func(t, a, b, c, d):
    return a * t**3 + b * t**2 + c * t + d

# Read data from a file
with open('hrn1', 'r') as f:
    lines = f.readlines()
set1 = [float(line.split()[3]) for line in lines if line.split()[0] == "Epoch:"]

with open('ffn1', 'r') as f:
    lines = f.readlines()
set2 = [float(line.split()[3]) for line in lines if line.split()[0] == "Epoch:"]

with open('trans', 'r') as f:
    lines = f.readlines()
set3 = [float(line.split()[3]) for line in lines if line.split()[0] == "Epoch:"]

times1 = np.arange(0, len(set1))
times2 = np.arange(0, len(set2))
times3 = np.arange(0, len(set3))
set1 = np.array(set1)
set2 = np.array(set2)
set3 = np.array(set3)

# Fit the data to the cubic function
params1, _ = curve_fit(cubic_func, times1, set1)
params2, _ = curve_fit(cubic_func, times2, set2)

# Generate the fitted curves using the cubic function
fit1 = cubic_func(times1, *params1)
fit2 = cubic_func(times2, *params2)

# Plotting
plt.figure()
plt.scatter(times1, set1, label="Hebbian", color='blue', s= 3)
plt.scatter(times2, set2, label="Feedforward", color='red', s= 3)
plt.scatter(times3, set3, label="Transformer", color='green', s= 3)
#plt.plot(times1, fit1, 'g--', label="Hebbian Fit", linewidth=2)
#plt.plot(times2, fit2, 'p--', label="FFN Fit", linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

