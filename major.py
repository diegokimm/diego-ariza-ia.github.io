import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('C:/Users/Usuario/Desktop/AI Diegokimm/Proyecto/majors2.csv')
df['time'] = pd.to_timedelta(df['time']).dt.total_seconds() / 60
#Visualizar cantidad de ganadores de las maratones en toda la historia
sns.histplot(df['marathon'], bins=1, kde=False)
plt.show()

# Se eliminan los datos anteriores a 1980 ya que sólo Boston tiene registros.
df = df[df['year'] > 1980]
# Graficar la evolución de los tiempos por maratón
plt.figure(figsize=(12, 6))
for marathon in df['marathon'].unique():
    subset = df[df['marathon'] == marathon]
    plt.plot(subset['year'], subset['time'], marker='o', label=marathon)

plt.xlabel("Año")
plt.ylabel("Tiempo (minutos)")
plt.title("Evolución de los tiempos en Maratones")
plt.legend()
plt.grid(True)
plt.show()


# Generar gráficas individuales de cada carrera por género

# Lista de maratones y géneros
maratones = df['marathon'].unique().tolist()
generos = ["Male", "Female"]

# Función para generar la gráfica
def generar_grafica():
    maraton_seleccionada = combo_maraton.get()
    genero_seleccionado = combo_genero.get()

    # Filtrar datos según selección
    df_filtrado = df[(df['marathon'] == maraton_seleccionada) & (df['gender'] == genero_seleccionado)].copy()

    # Crear la gráfica
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_filtrado, x='year', y='time', marker='o')

    # Personalizar la gráfica
    plt.xlabel("Año")
    plt.ylabel("Tiempo del ganador (minutos)")
    plt.title(f"Evolución del tiempo ganador en {maraton_seleccionada} ({genero_seleccionado})")
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

# Crear la ventana principal
root = tk.Tk()
root.title("Análisis de Maratones")

# Crear y posicionar etiquetas y combobox
tk.Label(root, text="Selecciona la maratón:").pack(pady=5)
combo_maraton = ttk.Combobox(root, values=maratones)
combo_maraton.pack(pady=5)
combo_maraton.current(0)

tk.Label(root, text="Selecciona el género:").pack(pady=5)
combo_genero = ttk.Combobox(root, values=generos)
combo_genero.pack(pady=5)
combo_genero.current(0)

# Botón para generar la gráfica
btn_generar = tk.Button(root, text="Generar Gráfica", command=generar_grafica)
btn_generar.pack(pady=20)

# Ejecutar la aplicación
root.mainloop()

"""
# Modelo de regresión lineal

maratones = df['marathon'].unique().tolist()
generos = ["Male", "Female"]

# Umbral de tiempo objetivo
tiempo_objetivo = 120

# Diccionario para almacenar predicciones
predicciones = {}

# Iterar sobre cada maratón y género
for maraton in maratones:
    for genero in generos:
        # Filtrar datos
        df_filtrado = df[(df['marathon'] == maraton) & (df['gender'] == genero)].copy()

        # Verificar si hay datos suficientes
        if df_filtrado.shape[0] < 2:
            continue

        # Extraer valores de año y tiempo
        X = df_filtrado['year'].values.reshape(-1, 1)  # Años (variable independiente)
        y = df_filtrado['time'].values  # Tiempos (variable dependiente)

        # Ajustar modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X, y)

        # Predecir el año en el que se alcance el tiempo menor a 120 minutos
        año_predicho = (tiempo_objetivo - modelo.intercept_) / modelo.coef_[0]

        # Guardar predicción si el año es razonable (futuro)
        if año_predicho > max(X):
            predicciones[(maraton, genero)] = int(año_predicho)

# Mostrar resultados
for (maraton, genero), año in predicciones.items():
    print(f"En la maratón {maraton} ({genero}), se espera que se alcance un tiempo menor a {tiempo_objetivo} min en el año {año}.")
"""



# Modelo de regresión lineal

# Umbral de tiempo objetivo
tiempo_objetivo = 120

# Diccionario para almacenar predicciones
predicciones = {}

# Iterar sobre cada maratón y género
for maraton in maratones:
    for genero in generos:
        # Filtrar datos
        df_filtrado = df[(df['marathon'] == maraton) & (df['gender'] == genero)].copy()
        
        # Verificar si hay datos suficientes
        if df_filtrado.shape[0] < 2:
            continue
        
        # Extraer valores de año y tiempo
        X = df_filtrado['year'].values.reshape(-1, 1)  # Años (variable independiente)
        y = df_filtrado['time'].values  # Tiempos (variable dependiente)
        
        # Ajustar modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Predecir el año en el que se alcanzará el tiempo menor a 120 minutos
        año_predicho = (tiempo_objetivo - modelo.intercept_) / modelo.coef_[0]


        # Evaluar el modelo
        y_pred = modelo.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Imprimir el comportamiento del modelo
        print(f"Comportamiento del modelo para la maratón {maraton} ({genero}):")
        print(f"  - Coeficiente de determinación (R²): {r2:.4f}")
        #print(f"  - Error cuadrático medio (MSE): {mse:.4f}")       



        # Condición si el año es razonable (futuro)
        if año_predicho > 2100:
            predicciones[(maraton, genero)] = "No se alcanzará el objetivo en un futuro razonable."
        elif año_predicho > max(X):  # Asegurarse de que el año predicho sea mayor al año actual máximo
            predicciones[(maraton, genero)] = int(año_predicho)

# Mostrar resultados
for (maraton, genero), resultado in predicciones.items():
    print(f"En la maratón {maraton} ({genero}), se espera que se alcance un tiempo menor a {tiempo_objetivo} min en el año {resultado}.")



    # Analisis gráfico de las predicciones
predicciones = {}

# Iterar sobre cada maratón y género
for maraton in maratones:
    for genero in generos:
        # Filtrar datos
        df_filtrado = df[(df['marathon'] == maraton) & (df['gender'] == genero)].copy()

        # Verificar si hay datos suficientes
        if df_filtrado.shape[0] < 2:
            continue

        # Extraer valores de año y tiempo
        X = df_filtrado['year'].values.reshape(-1, 1)  # Años (variable independiente)
        y = df_filtrado['time'].values  # Tiempos (variable dependiente)

        # Ajustar modelo de regresión lineal
        modelo = LinearRegression()
        modelo.fit(X, y)

        # Predecir el año en el que se alcance el tiempo menor a 120 minutos
        año_predicho = (tiempo_objetivo - modelo.intercept_) / modelo.coef_[0]

        # Guardar predicción si el año es razonable (futuro)
        if año_predicho > max(X):
            predicciones[(maraton, genero)] = int(año_predicho)

            # Generar valores de predicción para graficar la tendencia
            X_futuro = np.arange(min(X), int(año_predicho) + 10).reshape(-1, 1)
            y_futuro_pred = modelo.predict(X_futuro)

            # Graficar tendencia de tiempos con predicción
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=X.flatten(), y=y, label="Datos reales")
            plt.plot(X_futuro, y_futuro_pred, color='red', label=f'Regresión Lineal')
            plt.axhline(y=tiempo_objetivo, color='green', linestyle='--', label=f'Objetivo {tiempo_objetivo} min')
            plt.axvline(x=año_predicho, color='blue', linestyle='--', label=f'Predicción: {int(año_predicho)}')
            plt.xlabel("Año")
            plt.ylabel("Tiempo del ganador (min)")
            plt.title(f"{maraton} ({genero}) - Proyección Regresión Lineal")
            plt.legend()
            plt.grid(True)
            plt.show()

# Mostrar predicciones en texto
for (maraton, genero), año in predicciones.items():
    print(f"En la maratón {maraton} ({genero}), se espera que se alcance un tiempo menor a {tiempo_objetivo} min en el año {año}.")