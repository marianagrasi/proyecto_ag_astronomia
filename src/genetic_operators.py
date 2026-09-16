#1.funciones que manipulan los cromosomas.
import numpy as np
# Operadores para Cromosomas Binarios (Selección de Características) 
def seleccion_torneo(poblacion, fitness, k=3):
    """
    Selecciona un individuo usando el método de torneo.
    Elige 'k' individuos al azar y devuelve el que tiene el mejor fitness.
    """
    indices_participantes = np.random.choice(len(poblacion), k, replace=False)
    mejor_indice = indices_participantes[0]
    
    for idx in indices_participantes:
        if fitness[idx] > fitness[mejor_indice]:
            mejor_indice = idx
            
    return poblacion[mejor_indice]

def cruce_un_punto(padre1, padre2):
    """
    Realiza un cruce de un punto entre dos cromosomas binarios.
    Corta los padres en un punto aleatorio y los mezcla.
    """
    punto_cruce = np.random.randint(1, len(padre1))
    hijo1 = np.concatenate((padre1[:punto_cruce], padre2[punto_cruce:]))
    hijo2 = np.concatenate((padre2[:punto_cruce], padre1[punto_cruce:]))
    return hijo1, hijo2

def mutacion_bit_flip(cromosoma, prob_mutacion):
    """
    Aplica mutación bit-flip a un cromosoma binario.
    Cada gen tiene una 'prob_mutacion' de cambiar de 0 a 1 o viceversa.
    """
    for i in range(len(cromosoma)):
        if np.random.rand() < prob_mutacion:
            cromosoma[i] = 1 - cromosoma[i] # Cambia 0 a 1 y 1 a 0
    return cromosoma

# --- Operadores para Cromosomas de Valores Reales (Regresión, Clustering) ---

def seleccion_ruleta(poblacion, fitness):
    """
    Selecciona un individuo usando el método de ruleta.
    La probabilidad de selección es proporcional al fitness del individuo.
    """
    # Para evitar problemas con fitness negativos, ajustamos los valores
    fitness_ajustado = fitness - np.min(fitness) + 1e-9
    suma_fitness = np.sum(fitness_ajustado)
    
    if suma_fitness == 0:
        # Si todos los fitness son 0, seleccionamos al azar
        return poblacion[np.random.randint(len(poblacion))]
        
    prob_seleccion = fitness_ajustado / suma_fitness
    idx_seleccionado = np.random.choice(len(poblacion), p=prob_seleccion)
    return poblacion[idx_seleccionado]

def cruce_aritmetico(padre1, padre2, alpha=0.5):
    """
    Realiza un cruce aritmético (blend crossover).
    Los hijos son una combinación ponderada de los padres.
    """
    hijo1 = alpha * padre1 + (1 - alpha) * padre2
    hijo2 = alpha * padre2 + (1 - alpha) * padre1
    return hijo1, hijo2

def mutacion_gaussiana(cromosoma, prob_mutacion, sigma=0.1):
    """
    Aplica mutación gaussiana a un cromosoma de valor real.
    Añade un pequeño valor aleatorio (distribución normal) a cada gen.
    """
    for i in range(len(cromosoma)):
        if np.random.rand() < prob_mutacion:
            cromosoma[i] += np.random.normal(0, sigma)
    return cromosoma