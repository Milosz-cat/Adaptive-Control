import numpy as np
import time
from numpy.random import normal
import matplotlib.pyplot as plt


def printValues():
    print(f"Y1n: {Y1n}")
    print(f"Y2n: {Y2n}")
    print(f"W1n: {W1n}")
    print(f"W2n: {W2n}")


a1, a2 = 1 / 2, 1 / 4
b1, b2 = 1, 1
# z1 = np.random.uniform(-0.1, 0.1)
# z2 = np.random.uniform(-0.1, 0.1)
z1 = 0
z2 = 0

A = np.array([[a1, 0], [0, a2]])

B = np.array([[b1, 0], [0, b2]])

H = np.array([[0, 1], [1, 0]])

I = np.array([[1, 0], [0, 1]])

z = np.array([[z1], [z2]])

rng = 6

Y1n = np.zeros((1, rng))
W1n = np.zeros((2, rng))

Y2n = np.zeros((1, rng))
W2n = np.zeros((2, rng))

wynik1 = np.zeros((1, 2))
wynik2 = np.zeros((1, 2))

for i in range(rng):
    u1 = np.random.randint(1, 15)
    u2 = np.random.randint(1, 15)
    u = np.array([[u1], [u2]])

    y = (np.linalg.inv(I - (A @ H)) @ B @ u) + (np.linalg.inv(I - (A @ H)) @ z)

    Y1n[0, i] = y[0, 0]
    Y2n[0, i] = y[1, 0]
    W1n[0, i] = u1
    """
        1x2 @ 2x1 => 1x1
    """
    wynik1 = H[0, :] @ y
    W1n[1, i] = wynik1[0]
    W2n[0, i] = u2
    wynik2 = H[1, :] @ y
    W2n[1, i] = wynik2[0]


estymator1 = Y1n @ np.transpose(W1n) @ (np.linalg.inv(W1n @ np.transpose(W1n)))
estymator2 = Y2n @ np.transpose(W2n) @ (np.linalg.inv(W2n @ np.transpose(W2n)))

a1_est = estymator1[0, 1]
b1_est = estymator1[0, 0]
a2_est = estymator2[0, 1]
b2_est = estymator2[0, 0]

# a1_est = 1/2
# a2_est = 1/4
# b1_est = 1
# b2_est = 1

y1_zad = 4
y2_zad = 4

"""
    u = B^-1 * (I-A*H) * y - B^-1 *z
    2x2 * 2x2 -> 2x2; 2x2*2x1 -> 2x1 - 2x1
"""

A = np.array(([a1_est, 0], [0, a2_est]))

B = np.array(([b1_est, 0], [0, b2_est]))

z = np.array(([z1], [z2]))

y_zad = np.array([[y1_zad], [y2_zad]])


def find_optimal_u_clipped_binary(A, B, H, y_zad, z, max_time):
    start_time = time.time()
    u1_min, u1_max = -1, 1
    u2_min, u2_max = -1, 1
    min_Q = float("inf")
    optimal_u1, optimal_u2 = 0, 0

    while time.time() - start_time < max_time:
        u1_mid = (u1_min + u1_max) / 2
        u2_mid = (u2_min + u2_max) / 2

        for u1 in [u1_min, u1_mid, u1_max]:
            for u2 in [u2_min, u2_mid, u2_max]:
                if u1**2 + u2**2 <= 1:
                    u = np.array([[u1], [u2]])
                    y = (np.linalg.inv(I - (A @ H)) @ B @ u) + (
                        np.linalg.inv(I - (A @ H)) @ z
                    )
                    Q = ((y[0, 0] - y_zad[0, 0]) ** 2) + ((y[1, 0] - y_zad[1, 0]) ** 2)
                    if Q < min_Q:
                        min_Q = Q
                        optimal_u1, optimal_u2 = u1, u2

        # Aktualizacja zakresów w oparciu o wyniki obliczeń
        if optimal_u1 >= u1_mid:
            u1_min = u1_mid
        else:
            u1_max = u1_mid

        if optimal_u2 >= u2_mid:
            u2_min = u2_mid
        else:
            u2_max = u2_mid

    return optimal_u1, optimal_u2, min_Q


# docelowe u_szuk (u1, u2)
u_szuk = np.linalg.inv(B) @ (I - (A @ H)) @ y_zad - np.linalg.inv(B) @ z


# granice u1 i u2 z wzoru u1^2 + u2^2 <= 1
squared_norm_u_szuk = np.sum(u_szuk**2)
if squared_norm_u_szuk > 1:
    max_time = 10
    optimal_u1_clipped, optimal_u2_clipped, min_Q = find_optimal_u_clipped_binary(
        A, B, H, y_zad, z, max_time
    )
    u_szuk_clipped = np.array([[optimal_u1_clipped], [optimal_u2_clipped]])
else:
    u_szuk_clipped = u_szuk
    min_Q = ((y[0, 0] - y1_zad) ** 2) + ((y[1, 0] - y2_zad) ** 2)


# y = (np.linalg.inv(I - (A @ H)) @ B @ u_szuk_clipped) + (np.linalg.inv(I - (A @ H)) @ z)

# # obliczanie kosztow Q
# # im dokladniejsza estymata, tym mniejsze Q
# Q = ((y[0, 0] - y1_zad) ** 2) + ((y[1, 0] - y2_zad) ** 2)

# print(f"Estymator 1: {estymator1}")
# print(f"Estymator 2: {estymator2}")
print(f"Szukane\n\tu1: {u_szuk[0, 0]}\n\tu2 {u_szuk[1, 0]}")
print(
    f"Szukane ograniczone\n\tu1: {u_szuk_clipped[0, 0]}\n\tu2: {u_szuk_clipped[1, 0]}"
)
print(f"Q: {min_Q}")


fig, ax = plt.subplots(figsize=(8, 8))
circle = plt.Circle((0, 0), 1, color="blue", fill=False)
ax.add_artist(circle)
ax.scatter(
    u_szuk[0, 0],
    u_szuk[1, 0],
    color="red",
    marker="o",
    label=f"Nieograniczone u: ({u_szuk[0, 0]:.2f}, {u_szuk[1, 0]:.2f})",
)
ax.scatter(
    optimal_u1_clipped,
    optimal_u2_clipped,
    color="green",
    marker="o",
    label=f"Ograniczone u: ({optimal_u1_clipped:.2f}, {optimal_u2_clipped:.2f})",
)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel("u1")
ax.set_ylabel("u2")
ax.set_title("Optymalne Rozwiązania u1 i u2")
ax.legend()
ax.grid(True)
plt.show()
