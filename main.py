import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Importar  código modular
from src.genetic_algorithm import ejecutar_ag
from src.fitness_functions import (
    fitness_seleccion_caracteristicas,
    fitness_optimizacion_hiperparametros,
    fitness_clustering
)

# --- Configuración ---
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Cargar datos
try:
    df = pd.read_csv('data/sdss_sample.csv')
    print("Dataset cargado exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró 'data/sdss_sample.csv'.")
    exit()

# --- 1. Selección de Características con AG ---
print("\n--- 1. EJECUTANDO AG PARA SELECCIÓN DE CARACTERÍSTICAS ---")

features_candidatas = ['u', 'g', 'r', 'i', 'z', 'redshift']
target_clasificacion = 'class'

X_clf = df[features_candidatas]
y_clf = df[target_clasificacion]

resultados_clf = ejecutar_ag(
    fitness_func=fitness_seleccion_caracteristicas,
    num_generaciones=15,
    tam_poblacion=20,
    num_genes=len(features_candidatas),
    tipo_cromosoma='binario',
    prob_mutacion=0.1,
    X=X_clf,
    y=y_clf,
    features=features_candidatas
)
pd.DataFrame({
    'generacion': range(1, len(resultados_clf['historial_mejor_fitness']) + 1),
    'mejor_fitness': resultados_clf['historial_mejor_fitness'],
    'fitness_promedio': resultados_clf['historial_fitness_promedio']
}).to_csv('outputs/historial_seleccion.csv', index=False)

mejor_cromosoma_clf = resultados_clf['mejor_solucion']
mejores_features = [features_candidatas[i] for i, gen in enumerate(mejor_cromosoma_clf) if gen == 1]
mejor_accuracy = resultados_clf['mejor_fitness']

print(f"\nMejor subconjunto de características: {mejores_features}")
print(f"Accuracy asociado: {mejor_accuracy:.4f}")

with open('outputs/resultados_seleccion.json', 'w') as f:
    json.dump({'mejores_features': mejores_features, 'accuracy': mejor_accuracy}, f, indent=4)

# Gráfica de evolución
plt.figure(figsize=(10, 6))
plt.plot(resultados_clf['historial_mejor_fitness'], label='Mejor Fitness (Accuracy)')
plt.plot(resultados_clf['historial_fitness_promedio'], label='Fitness Promedio')
plt.title('Evolución del Fitness - Selección de Características')
plt.xlabel('Generación')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('outputs/evolucion_seleccion.png')
plt.close()

# Matriz de confusión con el mejor subconjunto
X_mejor = df[mejores_features]
X_train, X_test, y_train, y_test = train_test_split(X_mejor, y_clf, test_size=0.3, random_state=42)
knn_final = KNeighborsClassifier(n_neighbors=5)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)
cm = confusion_matrix(y_test, y_pred_final, labels=knn_final.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_final.classes_)
disp.plot()
plt.title('Matriz de Confusión - Mejor Subconjunto')
plt.savefig('outputs/matriz_confusion.png')
plt.close()


# --- 2. Optimización de Hiperparámetros con AG ---
print("\n--- 2. EJECUTANDO AG PARA OPTIMIZACIÓN DE HIPERPARÁMETROS ---")

features_reg = ['u', 'g', 'r', 'i', 'z']
target_reg = 'redshift'

X_reg = df[features_reg]
y_reg = df[target_reg]

resultados_reg = ejecutar_ag(
    fitness_func=fitness_optimizacion_hiperparametros,
    num_generaciones=15,
    tam_poblacion=20,
    num_genes=1, # Cromosoma de 1 gen: el valor de alpha
    tipo_cromosoma='real',
    prob_mutacion=0.2,
    elitismo=True,
    X=X_reg,
    y=y_reg
)
pd.DataFrame({
    'generacion': range(1, len(resultados_reg['historial_mejor_fitness']) + 1),
    'mejor_fitness': resultados_reg['historial_mejor_fitness'],
    'fitness_promedio': resultados_reg['historial_fitness_promedio']
}).to_csv('outputs/historial_hiperparametros.csv', index=False)

mejor_alpha = abs(resultados_reg['mejor_solucion'][0])

# Calcular MSE y R2 con el mejor alpha
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
modelo_final = Ridge(alpha=mejor_alpha)
modelo_final.fit(X_train, y_train)
y_pred_final = modelo_final.predict(X_test)
mse_final = mean_squared_error(y_test, y_pred_final)
r2_final = r2_score(y_test, y_pred_final)

print(f"\nMejor valor de alpha encontrado: {mejor_alpha:.4f}")
print(f"MSE con el mejor alpha: {mse_final:.4f}")
print(f"Coeficiente R²: {r2_final:.4f}")

with open('outputs/resultados_hiperparametros.json', 'w') as f:
    json.dump({'mejor_alpha': mejor_alpha, 'mse': mse_final, 'r2': r2_final}, f, indent=4)

# Gráfica de convergencia
plt.figure(figsize=(10, 6))
plt.plot(resultados_reg['historial_mejor_fitness'], label='Mejor Fitness (1/MSE)')
plt.plot(resultados_reg['historial_fitness_promedio'], label='Fitness Promedio')
plt.title('Convergencia del Fitness - Optimización de Hiperparámetros')
plt.xlabel('Generación')
plt.ylabel('Fitness (1/MSE)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/evolucion_hiperparametros.png')
plt.close()


# --- 3. Optimización de Clustering con AG ---
print("\n--- 3. EJECUTANDO AG PARA CLUSTERING ---")

features_cluster = ['u', 'g', 'r', 'i', 'z']
X_cluster = df[features_cluster]

# Estandarizamos los datos para el clustering
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# El cromosoma tiene 3 centroides * 5 dimensiones = 15 genes
num_genes_cluster = 3 * len(features_cluster)

resultados_cluster = ejecutar_ag(
    fitness_func=fitness_clustering,
    num_generaciones=20,
    tam_poblacion=30,
    num_genes=num_genes_cluster,
    tipo_cromosoma='real',
    prob_mutacion=0.1,
    elitismo=True,
    X=pd.DataFrame(X_cluster_scaled, columns=features_cluster)
)
pd.DataFrame({
    'generacion': range(1, len(resultados_cluster['historial_mejor_fitness']) + 1),
    'mejor_fitness': resultados_cluster['historial_mejor_fitness'],
    'fitness_promedio': resultados_cluster['historial_fitness_promedio']
}).to_csv('outputs/historial_clustering.csv', index=False)

mejor_cromosoma_cluster = resultados_cluster['mejor_solucion']
centroides_ag = mejor_cromosoma_cluster.reshape(3, len(features_cluster))

print(f"\nMejor fitness de clustering (1/SSE): {resultados_cluster['mejor_fitness']:.6f}")

np.savetxt('outputs/centroides_ag.csv', centroides_ag, delimiter=',', header=','.join(features_cluster), comments='')

# Comparación con KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)
centroides_kmeans = kmeans.cluster_centers_

# Visualización de los clusters
plt.figure(figsize=(15, 5))

# Gráfica 1: Clusters del AG
plt.subplot(1, 3, 1)
plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1], c='gray', alpha=0.5, label='Datos')
plt.scatter(centroides_ag[:, 0], centroides_ag[:, 1], c='red', marker='X', s=200, label='Centroides AG')
plt.title('Clusters por AG (datos estandarizados)')
plt.xlabel('u (escalado)')
plt.ylabel('g (escalado)')
plt.legend()

# Gráfica 2: Clusters de KMeans
plt.subplot(1, 3, 2)
plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(centroides_kmeans[:, 0], centroides_kmeans[:, 1], c='red', marker='X', s=200, label='Centroides KMeans')
plt.title('Clusters por KMeans')
plt.xlabel('u (escalado)')
plt.ylabel('g (escalado)')
plt.legend()

# Gráfica 3: Clases reales
plt.subplot(1, 3, 3)
clases_numericas = pd.factorize(df[target_clasificacion])[0]
plt.scatter(X_cluster_scaled[:, 0], X_cluster_scaled[:, 1], c=clases_numericas, cmap='plasma', alpha=0.5)
plt.title('Clases Reales')
plt.xlabel('u (escalado)')
plt.ylabel('g (escalado)')

plt.tight_layout()
plt.savefig('outputs/comparacion_clusters.png')
plt.close()

# Gráfica de convergencia
plt.figure(figsize=(10, 6))
plt.plot(resultados_cluster['historial_mejor_fitness'], label='Mejor Fitness (1/SSE)')
plt.plot(resultados_cluster['historial_fitness_promedio'], label='Fitness Promedio')
plt.title('Convergencia del Fitness - Clustering')
plt.xlabel('Generación')
plt.ylabel('Fitness (1/SSE)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/evolucion_clustering.png')
plt.close()

print("\n--- PROCESO COMPLETADO ---")
