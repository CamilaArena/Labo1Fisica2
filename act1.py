import numpy as np
import plotly.graph_objects as go

# Constante de Coulomb
k = 8.99e9  # N·m²/C²

# Función para ingresar las cargas desde la consola
def ingresar_cargas():
    n = int(input("¿Cuántas cargas quieres ingresar? (mínimo 3): "))
    cargas = []
    for i in range(n):
        print(f"\nIngrese los datos de la carga {i+1}:")
        x = float(input("Posición en x: "))
        y = float(input("Posición en y: "))
        q = float(input("Magnitud de la carga (en Coulombs, con signo): "))
        cargas.append({"pos": np.array([x, y]), "q": q})
    return cargas

# Función para calcular el campo eléctrico en un punto
def campo_electrico(punto, carga):
    r = punto - carga["pos"]
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.array([0, 0])  # Evitar división por cero
    return k * carga["q"] / r_mag**3 * r  # E = k * q * r / |r|^3 (vectorial)

# Ingresar las cargas
cargas = ingresar_cargas()

# Ingresar el punto de referencia
print("\nIngrese el punto de referencia para el campo eléctrico:")
px = float(input("Posición en x: "))
py = float(input("Posición en y: "))
punto_referencia = np.array([px, py])

# Crear una malla de puntos para el campo eléctrico
x = np.linspace(-5, 5, 20)  # Reducir el rango de -10 a 10 a -5 a 5
y = np.linspace(-5, 5, 20)  # Reducir el rango de -10 a 10 a -5 a 5
X, Y = np.meshgrid(x, y)
U = np.zeros_like(X)
V = np.zeros_like(Y)

# Calcular el campo eléctrico en cada punto de la malla
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        punto = np.array([X[i, j], Y[i, j]])
        E = np.array([0.0, 0.0])
        for carga in cargas:
            E += campo_electrico(punto, carga)
        U[i, j] = E[0]
        V[i, j] = E[1]

# Imprimir el campo eléctrico en el punto de referencia
E_referencia = np.array([0.0, 0.0])
for carga in cargas:
    E_referencia += campo_electrico(punto_referencia, carga)
Ex_referencia = E_referencia[0]
Ey_referencia = E_referencia[1]
E_magnitud_referencia = np.linalg.norm(E_referencia)
print("\nCampo eléctrico en el punto de referencia ({}, {}):".format(px, py))
print("Componente Ex: {:.3e} N/C".format(Ex_referencia))
print("Componente Ey: {:.3e} N/C".format(Ey_referencia))
print("Magnitud del campo eléctrico: {:.3e} N/C".format(E_magnitud_referencia))

# Crear el gráfico
fig = go.Figure()

# Graficar las cargas
for carga in cargas:
    fig.add_trace(go.Scatter(
        x=[carga["pos"][0]],
        y=[carga["pos"][1]],
        mode='markers+text',
        marker=dict(size=12, color='red' if carga["q"] > 0 else 'blue'),  # Aumentar el tamaño de los marcadores
        text=[f'q={carga["q"]:.2e} C'],
        textposition='top center',
        name=f'Carga en ({carga["pos"][0]:.1f}, {carga["pos"][1]:.1f})'
    ))

# Graficar las líneas de campo como flechas
fig.add_trace(go.Scatter(
    x=X.flatten(),
    y=Y.flatten(),
    mode='markers',
    marker=dict(size=5, color='black'),
    name='Puntos de Malla'
))

# Graficar las flechas del campo eléctrico
fig.add_trace(go.Scatter(
    x=np.concatenate([X.flatten(), X.flatten() + 0.2*U.flatten()]),  # Reducir la longitud de las flechas
    y=np.concatenate([Y.flatten(), Y.flatten() + 0.2*V.flatten()]),  # Reducir la longitud de las flechas
    mode='lines',
    line=dict(width=2, color='blue'),
    name='Campo Eléctrico'
))

# Configurar el diseño del gráfico
fig.update_layout(
    title='Campo Eléctrico debido a las Cargas',
    xaxis_title='Posición en x',
    yaxis_title='Posición en y',
    showlegend=True,
    xaxis=dict(range=[-5, 5]),  # Reducir el rango del eje x
    yaxis=dict(range=[-5, 5]),  # Reducir el rango del eje y
    autosize=False,
    width=800,
    height=800
)

# Mostrar el gráfico
fig.show()
