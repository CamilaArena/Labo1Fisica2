import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Constante de Coulomb
k = 9e9

# Función para calcular el campo eléctrico en un punto dado por una carga puntual
def campo_electrico(q, r, r0):
    r_vec = r - r0
    r_mag = np.linalg.norm(r_vec)
    E = k * q * r_vec / r_mag**3
    return E

# Función para calcular el potencial eléctrico en un punto dado por una carga puntual
def potencial_electrico(q, r, r0):
    r_mag = np.linalg.norm(r - r0)
    V = k * q / r_mag
    return V

# Obtener los datos de las cargas por consola
num_cargas = int(input("Ingrese el número de cargas (mínimo 3): "))
cargas = []
for i in range(num_cargas):
    q = float(input(f"Ingrese la carga {i+1} (en C): "))
    x = float(input(f"Ingrese la coordenada x de la carga {i+1}: "))
    y = float(input(f"Ingrese la coordenada y de la carga {i+1}: "))
    cargas.append((q, np.array([x, y])))

# Obtener el punto de referencia por consola
x0 = float(input("Ingrese la coordenada x del punto de referencia: "))
y0 = float(input("Ingrese la coordenada y del punto de referencia: "))
r0 = np.array([x0, y0])

# Calcular el campo eléctrico resultante en el punto de referencia
E_total = np.zeros(2)
for q, r in cargas:
    E_total += campo_electrico(q, r, r0)

# Calcular el potencial eléctrico resultante en el punto de referencia
V_total = 0
for q, r in cargas:
    V_total += potencial_electrico(q, r, r0)

# Imprimir los resultados por consola
print("El campo eléctrico resultante en el punto (", x0, ",", y0, ") es:", E_total)
print("El potencial eléctrico resultante en el punto (", x0, ",", y0, ") es:", V_total)

# Crear una malla para graficar
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Calcular el campo eléctrico y el potencial eléctrico en cada punto de la malla
Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
V = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        r = np.array([x[i], y[j]])
        for q, r_q in cargas:
            E = campo_electrico(q, r_q, r)
            Ex[i, j] += E[0]
            Ey[i, j] += E[1]
            V[i, j] += potencial_electrico(q, r_q, r)

# Graficar las líneas de campo eléctrico
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ex, Ey, density=2, cmap='viridis')
plt.scatter([q[1][0] for q in cargas], [q[1][1] for q in cargas], color='red', label='Cargas')
plt.scatter(x0, y0, color='blue', label='Punto de referencia')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Líneas de campo eléctrico')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el potencial eléctrico
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, V, levels=20, cmap='viridis')
plt.colorbar()
plt.scatter([q[1][0] for q in cargas], [q[1][1] for q in cargas], color='red', label='Cargas')
plt.scatter(x0, y0, color='blue', label='Punto de referencia')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potencial eléctrico')
plt.legend()
plt.grid(True)
plt.show()