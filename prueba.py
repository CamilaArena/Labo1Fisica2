import numpy as np
import matplotlib.pyplot as plt

# Constante de Coulomb (en N·m²/C²)
k = 9e9

# Función para calcular el campo eléctrico en un punto dado por una carga infinitesimal
def campo_electrico_lineal(dq, r, r0):
    r_vec = r - r0  # Vector que une el punto de la carga con el punto donde se calcula el campo
    r_mag = np.linalg.norm(r_vec)  # Magnitud del vector r_vec (distancia)
    E = k * dq * r_vec / r_mag**3  # Fórmula del campo eléctrico
    return E

# Función para calcular el potencial eléctrico en un punto dado por una carga infinitesimal
def potencial_electrico_lineal(dq, r, r0):
    r_mag = np.linalg.norm(r - r0)  # Magnitud del vector distancia entre r y r0
    V = k * dq / r_mag  # Fórmula del potencial eléctrico
    return V

# Parámetros de la distribución continua de carga (línea de carga)
largo_linea = 10  # Longitud de la línea de carga (en metros)
densidad_carga = 1e-9  # Densidad lineal de carga (C/m)
num_elementos = 100  # Número de elementos para aproximar la distribución continua
x_linea = np.linspace(-largo_linea/2, largo_linea/2, num_elementos)  # Coordenadas x de la línea
y_linea = np.zeros(num_elementos)  # La línea de carga está sobre el eje x
cargas = [(densidad_carga * (largo_linea / num_elementos), np.array([x, 0])) for x in x_linea]  # Lista de cargas infinitesimales

# Crear una malla de puntos para graficar el campo y el potencial eléctricos
x = np.linspace(-15, 15, 200)  # Valores de x
y = np.linspace(-10, 10, 200)  # Valores de y
X, Y = np.meshgrid(x, y)  # Crear una malla (grid) de puntos

# Inicializar matrices para los componentes del campo eléctrico (Ex, Ey) y el potencial (V)
Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
V = np.zeros_like(X)

# Calcular el campo eléctrico y el potencial en cada punto de la malla
for i in range(len(x)):
    for j in range(len(y)):
        r = np.array([x[i], y[j]])  # Coordenada actual en la malla
        for dq, r_q in cargas:
            E = campo_electrico_lineal(dq, r_q, r)  # Calcular el campo eléctrico
            Ex[i, j] += E[0]  # Componente x del campo
            Ey[i, j] += E[1]  # Componente y del campo
            V[i, j] += potencial_electrico_lineal(dq, r_q, r)  # Calcular el potencial eléctrico

# Crear el gráfico combinado del campo eléctrico y el potencial
plt.figure(figsize=(10, 8))

# Graficar las líneas de campo eléctrico usando un diagrama de flujo
plt.streamplot(X, Y, Ex, Ey, color='black', linewidth=0.5, density=2)

# Graficar el potencial eléctrico usando un mapa de contornos
contour = plt.contourf(X, Y, V, levels=20, cmap='coolwarm', alpha=0.7)
plt.colorbar(contour, label='Potencial eléctrico (V)')

# Marcar la posición de la línea de carga
plt.plot(x_linea, y_linea, color='red', label='Línea de carga')

# Añadir etiquetas y leyenda
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Campo eléctrico y Potencial eléctrico para una Distribución Continua de Carga')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()