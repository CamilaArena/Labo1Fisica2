import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constantes
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
I = 1.0                 # Corriente en las bobinas (A)
a = 1.0                 # Radio de las espiras (m)
d = a                   # Distancia entre las espiras (m)

# Función para calcular el campo magnético de una espira (paralela al eje z)
def biot_savart_espira_z(x, y, z, x_offset, N=500):
    theta = np.linspace(0, 2 * np.pi, N)
    dtheta = 2 * np.pi / N

    Bx, By, Bz = 0, 0, 0
    for t in theta:
        r_prime = np.array([x_offset, a * np.cos(t), a * np.sin(t)])
        dl = np.array([0, -a * np.sin(t), a * np.cos(t)]) * dtheta
        r = np.array([x, y, z])
        r_diff = r - r_prime
        r_mag = np.linalg.norm(r_diff)
        if r_mag == 0:
            continue
        dB = (mu0 * I / (4 * np.pi)) * np.cross(dl, r_diff) / r_mag**3
        Bx += dB[0]
        By += dB[1]
        Bz += dB[2]

    return np.array([Bx, By, Bz])

# Generar el campo magnético combinado en 2D
x_range = np.linspace(-2, 2, 20)
z_range = np.linspace(-2, 2, 20)
y_plane = 0.0  # Plano medio entre las espiras
X, Z = np.meshgrid(x_range, z_range)

Bx_total = np.zeros(X.shape)
Bz_total = np.zeros(Z.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B_espira1 = biot_savart_espira_z(X[i, j], y_plane, Z[i, j], -d/2)
        B_espira2 = biot_savart_espira_z(X[i, j], y_plane, Z[i, j], d/2)
        B_total = B_espira1 + B_espira2
        Bx_total[i, j] = B_total[0]
        Bz_total[i, j] = B_total[2]

# Magnitud del campo magnético
B_magnitude = np.sqrt(Bx_total**2 + Bz_total**2)

# Graficar en 2D
plt.figure(figsize=(10, 8))
plt.streamplot(X, Z, Bx_total, Bz_total, color=B_magnitude, cmap='plasma', linewidth=1)
plt.colorbar(label='|B| (T)')

# Dibujar las espiras
theta = np.linspace(0, 2 * np.pi, 100)
z_circle = a * np.cos(theta)
y_circle = a * np.sin(theta)
plt.plot([-d/2] * len(theta), z_circle, color='orange', linewidth=2, label='Espira Izquierda (x = -a/2)')
plt.plot([d/2] * len(theta), z_circle, color='blue', linewidth=2, label='Espira Derecha (x = a/2)')

plt.title('Campo magnético de las bobinas de Helmholtz (Plano y=0)')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.legend(loc='upper right')
plt.grid()
plt.show(block=False)

# Generar el campo magnético combinado en 3D
x_range_3d = np.linspace(-2, 2, 10)
y_range_3d = np.linspace(-2, 2, 10)
z_range_3d = np.linspace(-2, 2, 10)
X3D, Y3D, Z3D = np.meshgrid(x_range_3d, y_range_3d, z_range_3d)

Bx_3D = np.zeros(X3D.shape)
By_3D = np.zeros(Y3D.shape)
Bz_3D = np.zeros(Z3D.shape)

for i in range(X3D.shape[0]):
    for j in range(X3D.shape[1]):
        for k in range(X3D.shape[2]):
            B_espira1 = biot_savart_espira_z(X3D[i, j, k], Y3D[i, j, k], Z3D[i, j, k], -d/2)
            B_espira2 = biot_savart_espira_z(X3D[i, j, k], Y3D[i, j, k], Z3D[i, j, k], d/2)
            B_total = B_espira1 + B_espira2
            Bx_3D[i, j, k] = B_total[0]
            By_3D[i, j, k] = B_total[1]
            Bz_3D[i, j, k] = B_total[2]

# Graficar en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Añadir vectores de campo magnético
ax.quiver(X3D, Y3D, Z3D, Bx_3D, By_3D, Bz_3D, length=0.2, normalize=True, color='blue', alpha=0.6)

# Añadir las espiras
ax.plot([-d/2] * len(theta), z_circle, y_circle, color='orange', linewidth=3, label='Espira Izquierda')
ax.plot([d/2] * len(theta), z_circle, y_circle, color='blue', linewidth=3, label='Espira Derecha')

# Ajustes del gráfico
ax.set_title('Campo magnético de las bobinas de Helmholtz en 3D')
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.set_zlabel('y (m)')
ax.legend()
plt.show()

# Calcular el campo magnético en un punto específico
x1, y1, z1 = 0.5, 0.5, 0.5
B_point = biot_savart_espira_z(x1, y1, z1, -d/2) + biot_savart_espira_z(x1, y1, z1, d/2)
print(f"Campo magnético en ({x1}, {y1}, {z1}): B = {B_point} T")
