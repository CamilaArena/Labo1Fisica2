import numpy as np
import matplotlib.pyplot as plt

# Constante de Coulomb (en N·m²/C²)
k = 9e9

# Función para calcular el campo eléctrico en un punto dado por una carga puntual
def campo_electrico(q, r, r0):
    r_vec = r - r0  # Vector que une la carga con el punto donde se calcula el campo
    r_mag = np.linalg.norm(r_vec)  # Magnitud del vector de distancia
    if r_mag == 0:
        return np.zeros(2)  # Evitar división por cero
    E = k * q * r_vec / r_mag**3  # Campo eléctrico
    return E

# Función para calcular el potencial eléctrico en un punto dado por una carga puntual
def potencial_electrico(q, r, r0):
    r_mag = np.linalg.norm(r - r0)  # Magnitud del vector distancia
    if r_mag == 0:
        return 0  # Evitar división por cero
    V = k * q / r_mag  # Potencial eléctrico
    return V

# Solicitar al usuario los datos de las cargas
def solicitar_datos_cargas():
    num_cargas = int(input("Ingrese el número de cargas (mínimo 3): "))
    cargas = []
    for i in range(num_cargas):
        q = float(input(f"Ingrese la carga {i+1} (en C): "))
        x = float(input(f"Ingrese la coordenada x de la carga {i+1}: "))
        y = float(input(f"Ingrese la coordenada y de la carga {i+1}: "))
        cargas.append((q, np.array([x, y])))
    return cargas

# Obtener el punto de referencia donde se calcularán el campo y el potencial eléctricos
def obtener_punto_referencia():
    x0 = float(input("Ingrese la coordenada x del punto de referencia: "))
    y0 = float(input("Ingrese la coordenada y del punto de referencia: "))
    return np.array([x0, y0])

# Función para calcular el campo eléctrico y el potencial en toda la malla
def calcular_campo_potencial(cargas, X, Y):
    Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
    V = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = np.array([X[i, j], Y[i, j]])
            for q, r_q in cargas:
                E = campo_electrico(q, r, r_q)
                Ex[i, j] += E[0]
                Ey[i, j] += E[1]
                V[i, j] += potencial_electrico(q, r, r_q)
    
    return Ex, Ey, V

# Función para graficar las líneas de campo y el potencial eléctrico
def graficar(Ex, Ey, V, cargas, x0, y0):
    # Graficar las líneas de campo eléctrico usando streamplot
    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, Ex, Ey, color=np.log(np.sqrt(Ex**2 + Ey**2)), cmap='inferno', linewidth=1.2, density=2)
    
    # Graficar las cargas con diferentes colores para positivas y negativas
    pos_legend = False  # Control para la leyenda de cargas positivas
    neg_legend = False  # Control para la leyenda de cargas negativas
    for q, pos in cargas:
        if q > 0:
            if not pos_legend:  # Solo añadir la leyenda una vez
                plt.scatter(pos[0], pos[1], color='red', edgecolor='black', s=200, marker='o', zorder=5, label='Carga positiva')
                pos_legend = True
            else:
                plt.scatter(pos[0], pos[1], color='red', edgecolor='black', s=200, marker='o', zorder=5)
        else:
            if not neg_legend:  # Solo añadir la leyenda una vez
                plt.scatter(pos[0], pos[1], color='blue', edgecolor='black', s=200, marker='o', zorder=5, label='Carga negativa')
                neg_legend = True
            else:
                plt.scatter(pos[0], pos[1], color='blue', edgecolor='black', s=200, marker='o', zorder=5)
                
    plt.scatter(x0, y0, color='green', label='Punto de referencia', zorder=6)
    plt.xlim(-10, 15)
    plt.ylim(-10, 15)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Líneas de Campo Eléctrico')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')  # Fondo blanco
    plt.show(block=False)  # Mostrar la gráfica sin bloquear la ejecución

    # Graficar el potencial eléctrico usando un mapa de contornos
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, V, levels=20, cmap='viridis')
    plt.colorbar()
    
    # Graficar las cargas con diferentes colores
    for q, pos in cargas:
        if q > 0:
            plt.scatter(pos[0], pos[1], color='red', edgecolor='black', s=200, marker='o', zorder=5)
        else:
            plt.scatter(pos[0], pos[1], color='blue', edgecolor='black', s=200, marker='o', zorder=5)
    
    plt.scatter(x0, y0, color='green', label='Punto de referencia', zorder=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Potencial Eléctrico')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# --- Programa Principal ---
# Obtener los datos de las cargas
cargas = solicitar_datos_cargas()

# Obtener el punto de referencia
punto_referencia = obtener_punto_referencia()

# Calcular el campo eléctrico resultante en el punto de referencia
E_total = np.zeros(2)
V_total = 0

for q, r in cargas:
    E_total += campo_electrico(q, punto_referencia, r)
    V_total += potencial_electrico(q, punto_referencia, r)

# Imprimir los resultados por pantalla
print(f"El campo eléctrico resultante en el punto ({punto_referencia[0]}, {punto_referencia[1]}) es: {E_total}")
print(f"El potencial eléctrico resultante en el punto ({punto_referencia[0]}, {punto_referencia[1]}) es: {V_total}")

# Crear una malla de puntos para graficar el campo y el potencial eléctricos
x = np.linspace(-10, 15, 400)
y = np.linspace(-10, 15, 400)
X, Y = np.meshgrid(x, y)

# Calcular el campo eléctrico y el potencial en la malla
Ex, Ey, V = calcular_campo_potencial(cargas, X, Y)

# Graficar los resultados
graficar(Ex, Ey, V, cargas, punto_referencia[0], punto_referencia[1])
