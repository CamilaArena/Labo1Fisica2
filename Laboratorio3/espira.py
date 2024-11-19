import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Solicitar coordenadas al usuario
x_user = float(input("Ingrese la coordenada x del punto de observación (en metros): "))
y_user = float(input("Ingrese la coordenada y del punto de observación (en metros): "))
z_user = float(input("Ingrese la coordenada z del punto de observación (en metros): "))

# Constantes
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
I2 = 1.0                # Corriente en amperios
a = 1.0                 # Radio de la espira (m)

# Función para calcular el campo magnético en un punto debido a la espira
def biot_savart_espira(x, y, z, N=500):
    """
    Calcula el campo magnético en (x, y, z) debido a una espira circular de radio a y corriente I2.
    """
    theta = np.linspace(0, 2 * np.pi, N)  # Ángulos a lo largo de la espira
    dtheta = 2 * np.pi / N                # Diferencial de ángulo

    Bx, By, Bz = 0, 0, 0
    for t in theta:
        # Posición del elemento de corriente sobre la espira
        r_prime = np.array([0, a * np.cos(t), a * np.sin(t)])  # Coordenadas de la espira en el plano xz
        # Elemento diferencial de corriente
        dl = np.array([0, -a * np.sin(t), a * np.cos(t)]) * dtheta
        # Punto de observación
        r = np.array([x, y, z])
        r_diff = r - r_prime
        r_mag = np.linalg.norm(r_diff)
        if r_mag == 0:
            continue
        # Producto cruzado para el campo magnético
        dB = (mu0 * I2 / (4 * np.pi)) * np.cross(dl, r_diff) / r_mag**3
        Bx += dB[0]
        By += dB[1]
        Bz += dB[2]

    return np.array([Bx, By, Bz])

# Gráfica 2D del campo magnético
x_range = np.linspace(-2, 2, 20)
y_range = np.linspace(-2, 2, 20)
z_plane = z_user  # Plano basado en el valor ingresado
X, Y = np.meshgrid(x_range, y_range)

Bx_plane = np.zeros(X.shape)
By_plane = np.zeros(Y.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B = biot_savart_espira(X[i, j], Y[i, j], z_plane)
        Bx_plane[i, j] = B[0]
        By_plane[i, j] = B[1]

# Magnitud del campo magnético
B_magnitude = np.sqrt(Bx_plane**2 + By_plane**2)

# Graficar en 2D
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Bx_plane, By_plane, color=B_magnitude, cmap='viridis')
plt.colorbar(label='|B| (T)')
plt.plot([0, 0], [-a, a], color='red', linewidth=2, label='Espira')  # Representación de la espira en el eje Y
plt.title(f'Campo magnético en el plano z = {z_plane:.2f}')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.grid()
plt.show(block=False)

# Gráfica 3D del campo magnético
x_range_3d = np.linspace(-2, 2, 8)
y_range_3d = np.linspace(-2, 2, 8)
z_range_3d = np.linspace(-2, 2, 8)
X3D, Y3D, Z3D = np.meshgrid(x_range_3d, y_range_3d, z_range_3d)

Bx_3D = np.zeros(X3D.shape)
By_3D = np.zeros(Y3D.shape)
Bz_3D = np.zeros(Z3D.shape)

for i in range(X3D.shape[0]):
    for j in range(X3D.shape[1]):
        for k in range(X3D.shape[2]):
            B = biot_savart_espira(X3D[i, j, k], Y3D[i, j, k], Z3D[i, j, k])
            Bx_3D[i, j, k] = B[0]
            By_3D[i, j, k] = B[1]
            Bz_3D[i, j, k] = B[2]

# Normalizar las flechas para mejorar visualización
magnitude = np.sqrt(Bx_3D**2 + By_3D**2 + Bz_3D**2)
nonzero = magnitude > 1e-12  # Evitar divisiones por 0
Bx_3D[nonzero] /= magnitude[nonzero]
By_3D[nonzero] /= magnitude[nonzero]
Bz_3D[nonzero] /= magnitude[nonzero]

# Gráfica en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Añadir flechas del campo magnético
ax.quiver(X3D, Y3D, Z3D, Bx_3D, By_3D, Bz_3D, length=0.5, normalize=True, color='blue', alpha=0.8)

# Añadir la espira como un círculo
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = np.zeros_like(theta)
y_circle = a * np.cos(theta)
z_circle = a * np.sin(theta)
ax.plot(x_circle, y_circle, z_circle, color='red', linewidth=3, label='Espira')

# Etiquetas y título
ax.set_title('Campo magnético en 3D debido a una espira', fontsize=14)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.legend()
plt.show()

# Calcular el campo en un punto específico
B_point = biot_savart_espira(x_user, y_user, z_user)
print(f"Campo magnético en ({x_user}, {y_user}, {z_user}): B = {B_point} T")
