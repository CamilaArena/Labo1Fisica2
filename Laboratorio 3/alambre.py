import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constantes
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
I1 = 1.0                # Corriente en amperios
L = 2.0                 # Longitud del alambre (m)

# Función para calcular el campo magnético en un punto
def biot_savart(x, y, z, N=1000):
    """
    Calcula el campo magnético en (x, y, z) debido a un alambre recto de longitud L.
    """
    z_prime = np.linspace(-L/2, L/2, N)  # Posiciones a lo largo del alambre
    dz = L / N                           # Diferencial en z
    dl = np.array([0, 0, dz])            # Elemento de corriente diferencial

    Bx, By, Bz = 0, 0, 0
    for zp in z_prime:
        r_prime = np.array([0, 0, zp])   # Posición del elemento de corriente
        r = np.array([x, y, z])          # Punto de observación
        r_diff = r - r_prime
        r_mag = np.linalg.norm(r_diff)
        if r_mag == 0:
            continue
        dB = (mu0 * I1 / (4 * np.pi)) * np.cross(dl, r_diff) / r_mag**3
        Bx += dB[0]
        By += dB[1]
        Bz += dB[2]

    return np.array([Bx, By, Bz])

# Generar el campo magnético en un plano (2D gráfico)
x_range = np.linspace(-1, 1, 20)
y_range = np.linspace(-1, 1, 20)
z_plane = 0  # Plano de observación
X, Y = np.meshgrid(x_range, y_range)

Bx_plane = np.zeros(X.shape)
By_plane = np.zeros(Y.shape)
Bz_plane = np.zeros(Y.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B = biot_savart(X[i, j], Y[i, j], z_plane)
        Bx_plane[i, j] = B[0]
        By_plane[i, j] = B[1]
        Bz_plane[i, j] = B[2]

# Magnitud del campo magnético
B_magnitude = np.sqrt(Bx_plane**2 + By_plane**2 + Bz_plane**2)

# Graficar en 2D
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Bx_plane, By_plane, color=B_magnitude, cmap='viridis')
plt.colorbar(label='|B| (T)')
plt.scatter(0, 0, color='red', s=100, label='Alambre (eje z)')
plt.title('Campo magnético en el plano z = 0')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.grid()
plt.show(block=False)

# Generar el campo magnético en 3D
x_range_3d = np.linspace(-1, 1, 10)
y_range_3d = np.linspace(-1, 1, 10)
z_range_3d = np.linspace(-L, L, 10)
X3D, Y3D, Z3D = np.meshgrid(x_range_3d, y_range_3d, z_range_3d)

Bx_3D = np.zeros(X3D.shape)
By_3D = np.zeros(Y3D.shape)
Bz_3D = np.zeros(Z3D.shape)

for i in range(X3D.shape[0]):
    for j in range(X3D.shape[1]):
        for k in range(X3D.shape[2]):
            B = biot_savart(X3D[i, j, k], Y3D[i, j, k], Z3D[i, j, k])
            Bx_3D[i, j, k] = B[0]
            By_3D[i, j, k] = B[1]
            Bz_3D[i, j, k] = B[2]

# Graficar en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Añadir vectores de campo magnético
ax.quiver(X3D, Y3D, Z3D, Bx_3D, By_3D, Bz_3D, length=0.1, normalize=True, color='blue', alpha=0.6)

# Añadir alambre
z_line = np.linspace(-L/2, L/2, 100)
ax.plot([0] * len(z_line), [0] * len(z_line), z_line, color='red', linewidth=3, label='Alambre (eje z)')

ax.set_title('Campo magnético en 3D')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.legend()
plt.show()

# Calcular campo en un punto específico
x1, y1, z1 = 0.5, 0.5, 0.5  # Coordenadas del punto
B_point = biot_savart(x1, y1, z1)
print(f"Campo magnético en ({x1}, {y1}, {z1}): B = {B_point} T")
