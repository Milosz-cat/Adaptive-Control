# Revised code for generating and printing specific values of U, Z, and Y

import numpy as np
import random


# Function for generating rectangular disturbances
def Z_prostokatne(low, high):
    return random.uniform(low, high)


# Simulator function
def symulator(U=None, with_disturbance=True):
    if U is None:
        U = np.array([[5 * random.random()] for _ in range(2)])  # Random input signals
    else:
        U = np.array(U).reshape(2, 1)

    A = np.array([[0.5, 0], [0, 0.25]])  # System parameters
    B = np.array([[1, 0], [0, 1]])  # System parameters
    H = np.array([[0, 1], [1, 0]])  # Fixed matrix

    if with_disturbance:
        Z = np.array([[Z_prostokatne(-1, 1)] for _ in range(2)])  # Disturbances
    else:
        Z = np.array([[0], [0]])  # No disturbance

    Y = (
        np.linalg.inv(np.identity(2) - A @ H) @ B @ U
        + np.linalg.inv(np.identity(2) - A @ H) @ Z
    )
    return U.ravel(), Y.ravel(), Z.ravel()


# Random case
U_random, Y_random, Z_random = symulator()
print("Randomowe wartości:")
print("U:", U_random)
print("Y:", Y_random)
print("Z:", Z_random)

# Random case
U_random, Y_random, Z_random = symulator()
print("\nRandomowe wartości:")
print("U:", U_random)
print("Y:", Y_random)
print("Z:", Z_random)

# Specified case (u1=2, u2=3) without disturbances
U_specified, Y_specified, Z_specified = symulator([2, 3], with_disturbance=False)
print("\nGdy (u1=2, u2=3) bez zakłoceń:")
print("U:", U_specified)
print("Y:", Y_specified)
