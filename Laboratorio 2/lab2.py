import numpy as np
import matplotlib.pyplot as plt

nx, ny = 101, 161  # Tamaño de la grilla
tolerancia = 1e-3  # Tolerancia ε
max_iter = 10000   # Número máximo de iteraciones

# Inicializar el potencial con valores aleatorios entre -100 y 100
potencial = np.random.uniform(-100, 100, (nx, ny))

# Condiciones de Dirichlet en ciertos puntos de la grilla [fila,columna]
potencial[40, 80] = 100  # Borne 1 con potencial de 100V
potencial[30, 40] = -50  # Borne 2 con potencial de -50V
potencial[70, 120] = 75  # Borne 3 con potencial de 75V

# Método de relajación
def relajacion(potencial, max_iter, tolerancia):
    for it in range(max_iter):
        potencial_old = np.copy(potencial)
        # Actualizar el potencial usando el promedio de los vecinos
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                potencial[i, j] = 0.25 * (potencial_old[i+1, j] + potencial_old[i-1, j] + 
                                          potencial_old[i, j+1] + potencial_old[i, j-1])

        # Condiciones de Neumann: derivada cero en los bordes
        potencial[:, 0] = potencial[:, 1]
        potencial[:, -1] = potencial[:, -2]
        potencial[0, :] = potencial[1, :]
        potencial[-1, :] = potencial[-2, :]

        # Verificar convergencia
        cambio = np.max(np.abs(potencial - potencial_old))
        if cambio < tolerancia:
            print(f'Convergencia alcanzada en {it} iteraciones')
            break
    return potencial

# Resolver la ecuación de Laplace
potencial = relajacion(potencial, max_iter, tolerancia)

# Graficar las líneas equipotenciales
plt.figure(figsize=(8, 6))

vmin_pot = np.min(potencial)
vmax_pot = np.max(potencial)
print("Mínimo potencial:", np.min(potencial))
print("Máximo potencial:", np.max(potencial))
contour = plt.contourf(potencial.T, 100, cmap='viridis', vmin=vmin_pot, vmax=vmax_pot)  # Equipotenciales
plt.colorbar(contour, label='Potencial (V)')  


plt.title('Líneas Equipotenciales')
plt.xlabel('x')
plt.ylabel('y')
plt.show(block=False)  # Mostrar la gráfica sin bloquear la ejecución


# Gráfico de superficie 3D del potencial
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
ax.plot_surface(X, Y, potencial.T, cmap='viridis', edgecolor='none', vmin=vmin_pot, vmax=vmax_pot)
ax.set_title('Superficie del Potencial')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Potencial (V)')
plt.show()
