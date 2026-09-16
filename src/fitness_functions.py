#2.Funciones fitness
#definimos cómo medimos la "bondad" de un individuo para cda uno de los tres problemas.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error

# --- 1. Fitness para Selección de Características ---
def fitness_seleccion_caracteristicas(cromosoma, X, y, features, random_state=42):
    """
    Calcula la precisión (accuracy) de un KNN usando solo las características seleccionadas.
    """
    # Si el cromosoma no selecciona ninguna característica, el fitness es 0.
    if np.sum(cromosoma) == 0:
        return 0.0
    
    # Seleccionar las características activas (donde el gen es 1)
    caracteristicas_activas = [features[i] for i, gen in enumerate(cromosoma) if gen == 1]
    
    X_subset = X[caracteristicas_activas]
    
    # Partición de datos (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.3, random_state=random_state
    )
    
    # Modelo KNN y evaluación
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

# --- 2. Fitness para Optimización de Hiperparámetros ---
def fitness_optimizacion_hiperparametros(cromosoma, X, y, random_state=42):
    """
    Calcula el inverso del Error Cuadrático Medio (MSE) para un modelo Ridge.
    El cromosoma contiene un valor real para el hiperparámetro 'alpha'.
    """
    # El cromosoma[0] es el valor de alpha. Debe ser positivo.
    alpha = abs(cromosoma[0])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Modelo Ridge y evaluación
    modelo = Ridge(alpha=alpha)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    # Queremos maximizar el fitness, y el MSE es un error.
    # Devolvemos el inverso. Sumamos un pequeño epsilon para evitar división por cero.
    return 1.0 / (mse + 1e-6)

# --- 3. Fitness para Optimización de Clustering ---
def fitness_clustering(cromosoma, X):
    """
    Calcula el inverso de la Suma de Errores Cuadráticos (SSE) para 3 centroides.
    El cromosoma contiene las coordenadas de los 3 centroides.
    """
    n_features = X.shape[1]
    # El cromosoma tiene 3 * n_features valores. Lo reformamos a una matriz (3, n_features)
    centroides = cromosoma.reshape(3, n_features)
    
    sse = 0
    for _, punto in X.iterrows():
        # Encontrar el centroide más cercano
        distancias = [np.linalg.norm(punto.values - centroide) for centroide in centroides]
        idx_centroide_cercano = np.argmin(distancias)
        # Sumar la distancia al cuadrado
        sse += distancias[idx_centroide_cercano]**2
        
    # Queremos maximizar el fitness, y el SSE es un error. Devolvemos el inverso.
    return 1.0 / (sse + 1e-6)
