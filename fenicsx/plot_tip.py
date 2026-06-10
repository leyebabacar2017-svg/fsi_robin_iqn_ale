import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(
    "results/tip_history.txt"
)

t = data[:,0]
u = data[:,1]

plt.figure(figsize=(8,4))

plt.plot(t,u)

plt.grid(True)

plt.xlabel("Time (s)")
plt.ylabel("Tip displacement")

plt.tight_layout()

plt.show()