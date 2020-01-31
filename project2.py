import numpy as np
from math import inf
import matplotlib.pyplot as plt

# PARAMETERS ------------
L = 1
step = 100
V = 1
b = V/(L*step + 1)
maxIter = 10000
# -----------------------

# INITIALIZATION --------
X = np.arange(0, 2*L*step)
Y = np.arange(0, 2*L*step)

Z = np.zeros((2*L*step, 2*L*step))

for y in range(L*step + 1):
    Z[L*step-1, y] = b*y

for x in range(L*step, 2*L*step):
    Z[x, L*step] = b*(2*L*step - 1 - x)

for x in range(L*step, 2*L*step):
    for y in range(L*step):
        Z[x,y] = -inf

# -----------------------


# CODE ------------------
for counter in range(maxIter):
    maxE = -inf
    for x in X:
        for y in Y:
            # Check for bounds
            if x == L*step-1:
                if y < L*step + 1:
                    continue
            if y == L*step:
                if x >= L*step:
                    continue
            if x == 0 or y == 0 or x == 2*L*step -1 or y == 2*L*step-1:
                continue
            # If not on bounds -> update.
            prev = Z[x,y]
            Z[x,y] = 1/(4) * (Z[x-1, y] + Z[x + 1, y] + Z[x, y - 1] + Z[x, y + 1])
            # Record error
            if Z[x,y] != -inf and abs(prev- Z[x,y]) > maxE:
                maxE = abs(prev - Z[x,y])
    # Check for convergence
    if maxE < 10**(-5):
        break

# -----------------------


# PLOTTING --------------
fig = plt.figure()
ax = plt.gca()
im = ax.imshow(Z, cmap='jet', interpolation='nearest')
ax.contour(X,Y,Z)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Color Scale', rotation=-90, va="bottom")

fig.show()

# -----------------------
