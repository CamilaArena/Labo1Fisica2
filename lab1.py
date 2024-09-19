import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Constante de Coulomb (en N·m²/C²)
k = 9e9

# Función para calcular el campo eléctrico en un punto dado por una carga puntual
# q: magnitud de la carga (en C)
# r: posición donde se quiere calcular el campo eléctrico (en metros)
# r0: posición de la carga puntual (en metros)
def campo_electrico(q, r, r0):
    r_vec = r - r0  # Vector que une el punto de la carga con el punto donde se calcula el campo
    r_mag = np.linalg.norm(r_vec)  # Magnitud del vector r_vec (distancia)
    if r_mag == 0:
        return np.zeros(2)  # Si la distancia es cero, el campo es indefinido, retornamos [0, 0]
    E = k * q * r_vec / r_mag**3  # Fórmula del campo eléctrico
    return E

# Función para calcular el potencial eléctrico en un punto dado por una carga puntual
# q: magnitud de la carga (en C)
# r: posición donde se quiere calcular el potencial (en metros)
# r0: posición de la carga puntual (en metros)
def potencial_electrico(q, r, r0):
    r_mag = np.linalg.norm(r - r0)  # Magnitud del vector distancia entre r y r0
    if r_mag == 0:
        return 0  # Si la distancia es cero, el potencial es indefinido, retornamos 0
    
    V = k * q / r_mag  # Fórmula del potencial eléctrico
    return V

# Obtener los datos de las cargas por consola
num_cargas = int(input("Ingrese el número de cargas (mínimo 3): "))
cargas = []
# Solicitar las coordenadas y valores de las cargas al usuario
for i in range(num_cargas):
    q = float(input(f"Ingrese la carga {i+1} (en C): "))  # Valor de la carga
    x = float(input(f"Ingrese la coordenada x de la carga {i+1}: "))  # Coordenada x de la carga
    y = float(input(f"Ingrese la coordenada y de la carga {i+1}: "))  # Coordenada y de la carga
    cargas.append((q, np.array([x, y])))  # Guardar carga y su posición como tupla

# Obtener el punto de referencia donde se calcularán el campo y el potencial eléctricos
x0 = float(input("Ingrese la coordenada x del punto de referencia: "))
y0 = float(input("Ingrese la coordenada y del punto de referencia: "))
r0 = np.array([x0, y0])  # Punto de referencia (posición)

# Calcular el campo eléctrico resultante en el punto de referencia
E_total = np.zeros(2)  # Inicializar el campo eléctrico total en el punto (x0, y0)
for q, r in cargas:
    E_total += campo_electrico(q, r, r0)  # Sumar el campo eléctrico de cada carga en el punto

# Calcular el potencial eléctrico resultante en el punto de referencia
V_total = 0  # Inicializar el potencial eléctrico total en el punto (x0, y0)
for q, r in cargas:
    V_total += potencial_electrico(q, r, r0)  # Sumar el potencial de cada carga en el punto

# Imprimir los resultados por consola
print("El campo eléctrico resultante en el punto (", x0, ",", y0, ") es:", E_total)
print("El potencial eléctrico resultante en el punto (", x0, ",", y0, ") es:", V_total)

# Crear una malla de puntos para graficar el campo y el potencial eléctricos
x = np.linspace(-5, 5, 100)  # Valores de x entre -5 y 5
y = np.linspace(-5, 5, 100)  # Valores de y entre -5 y 5
X, Y = np.meshgrid(x, y)  # Crear una malla (grid) de puntos

# Inicializar matrices para los componentes del campo eléctrico (Ex, Ey) y el potencial (V)
Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
V = np.zeros_like(X)

# Calcular el campo eléctrico y el potencial en cada punto de la malla
for i in range(len(x)):
    for j in range(len(y)):
        r = np.array([x[i], y[j]])  # Coordenada actual en la malla
        for q, r_q in cargas:
            E = campo_electrico(q, r_q, r)  # Calcular el campo eléctrico en el punto (x[i], y[j])
            Ex[i, j] += E[0]  # Componente x del campo
            Ey[i, j] += E[1]  # Componente y del campo
            V[i, j] += potencial_electrico(q, r_q, r)  # Sumar el potencial eléctrico en el punto

# Graficar las líneas de campo eléctrico usando un diagrama de flujo
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ex, Ey, density=2, cmap='viridis')  # Graficar las líneas de campo
plt.scatter([q[1][0] for q in cargas], [q[1][1] for q in cargas], color='red', label='Cargas')  # Marcar las cargas
plt.scatter(x0, y0, color='blue', label='Punto de referencia')  # Marcar el punto de referencia
plt.xlabel('x')
plt.ylabel('y')
plt.title('Líneas de campo eléctrico')
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar la cuadrícula
plt.show(block=False)  # Mostrar la gráfica sin bloquear la ejecución

# Graficar el potencial eléctrico usando un mapa de contornos
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, V, levels=20, cmap='viridis')  # Graficar el potencial en diferentes niveles
plt.colorbar()  # Mostrar la barra de colores
plt.scatter([q[1][0] for q in cargas], [q[1][1] for q in cargas], color='red', label='Cargas')  # Marcar las cargas
plt.scatter(x0, y0, color='blue', label='Punto de referencia')  # Marcar el punto de referencia
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potencial eléctrico')
plt.legend()  # Mostrar la leyenda
plt.grid(True)  # Mostrar la cuadrícula
plt.show()  # Mostrar la segunda gráfica