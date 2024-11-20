import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constantes
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
I_alambre = 1.0         # Corriente en el alambre (A)
I_espira = 1.0          # Corriente en la espira (A)
L = 4.0                 # Longitud del alambre (m), ajustada para ser más larga que la espira
a = 0.5                 # Radio de la espira reducido (m)

# Función para calcular el campo magnético por un alambre recto
def biot_savart_alambre(x, y, z, N=1000):
    z_prime = np.linspace(-L/2, L/2, N)
    dz = L / N
    dl = np.array([0, 0, dz])

    Bx, By, Bz = 0, 0, 0
    for zp in z_prime:
        r_prime = np.array([0, 0, zp])
        r = np.array([x, y, z])
        r_diff = r - r_prime
        r_mag = np.linalg.norm(r_diff)
        if r_mag == 0:
            continue
        dB = (mu0 * I_alambre / (4 * np.pi)) * np.cross(dl, r_diff) / r_mag**3
        Bx += dB[0]
        By += dB[1]
        Bz += dB[2]

    return np.array([Bx, By, Bz])

# Función para calcular el campo magnético por una espira circular
def biot_savart_espira(x, y, z, N=500):
    theta = np.linspace(0, 2 * np.pi, N)
    dtheta = 2 * np.pi / N

    Bx, By, Bz = 0, 0, 0
    for t in theta:
        r_prime = np.array([a * np.cos(t), a * np.sin(t), 0])
        dl = np.array([-a * np.sin(t), a * np.cos(t), 0]) * dtheta
        r = np.array([x, y, z])
        r_diff = r - r_prime
        r_mag = np.linalg.norm(r_diff)
        if r_mag == 0:
            continue
        dB = (mu0 * I_espira / (4 * np.pi)) * np.cross(dl, r_diff) / r_mag**3
        Bx += dB[0]
        By += dB[1]
        Bz += dB[2]

    return np.array([Bx, By, Bz])

# Generar el campo magnético combinado en 3D
x_range_3d = np.linspace(-2, 2, 8)  # Más puntos para más flechas
y_range_3d = np.linspace(-2, 2, 8)
z_range_3d = np.linspace(-2, 2, 8)
X3D, Y3D, Z3D = np.meshgrid(x_range_3d, y_range_3d, z_range_3d)

Bx_3D = np.zeros(X3D.shape)
By_3D = np.zeros(Y3D.shape)
Bz_3D = np.zeros(Z3D.shape)

for i in range(X3D.shape[0]):
    for j in range(X3D.shape[1]):
        for k in range(X3D.shape[2]):
            B_alambre = biot_savart_alambre(X3D[i, j, k], Y3D[i, j, k], Z3D[i, j, k])
            B_espira = biot_savart_espira(X3D[i, j, k], Y3D[i, j, k], Z3D[i, j, k])
            B_total = B_alambre + B_espira
            Bx_3D[i, j, k] = B_total[0]
            By_3D[i, j, k] = B_total[1]
            Bz_3D[i, j, k] = B_total[2]

# Graficar en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Añadir vectores de campo magnético con flechas más largas
ax.quiver(X3D, Y3D, Z3D, Bx_3D, By_3D, Bz_3D, length=0.3, normalize=True, color='blue', alpha=0.7)

# Añadir la espira
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = a * np.cos(theta)
z_circle = a * np.sin(theta)
ax.plot(x_circle, [0] * len(theta), z_circle, color='orange', linewidth=3, label='Espira')

# Añadir el alambre
z_line = np.linspace(-L/2, L/2, 100)
ax.plot([0] * len(z_line), [0] * len(z_line), z_line, color='red', linewidth=3, label='Alambre')

# Ajustes del gráfico
ax.set_title('Campo magnético combinado en 3D')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.legend()
plt.show()
