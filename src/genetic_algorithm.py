#3.ciclo principal que ejecuta la evolución.
#Motor del Algoritmo Genético - ciclo principal que ejecuta la evolución.
import numpy as np
from .genetic_operators import *

def ejecutar_ag(fitness_func, num_generaciones, tam_poblacion, num_genes, 
                tipo_cromosoma='binario', prob_mutacion=0.1, 
                elitismo=True, **kwargs_fitness):
    """
    Función principal que ejecuta el Algoritmo Genético.
    """
    
    # 1. Inicialización de la población
    if tipo_cromosoma == 'binario':
        poblacion = [np.random.randint(0, 2, num_genes) for _ in range(tam_poblacion)]
    else: # 'real'
        # Inicializamos con valores aleatorios, ej. entre -1 y 1
        poblacion = [np.random.rand(num_genes) * 2 - 1 for _ in range(tam_poblacion)]

    historial_mejor_fitness = []
    historial_fitness_promedio = []
    historial_diversidad = []

    for gen in range(num_generaciones):
        # 2. Evaluación de la población
        fitness_poblacion = [fitness_func(ind, **kwargs_fitness) for ind in poblacion]
        
        # Guardar estadísticas
        mejor_fitness_gen = max(fitness_poblacion)
        historial_mejor_fitness.append(mejor_fitness_gen)
        historial_fitness_promedio.append(np.mean(fitness_poblacion))
        
        # Calcular diversidad (desviación estándar de los genes de toda la población)
        diversidad = np.mean([np.std(ind) for ind in poblacion])
        historial_diversidad.append(diversidad)
        
        print(f"Generación {gen+1}/{num_generaciones} | Mejor Fitness: {mejor_fitness_gen:.6f} | Fitness Promedio: {np.mean(fitness_poblacion):.6f}")

        # 3. Crear la nueva población
        nueva_poblacion = []
        
        # Elitismo: conservar el mejor individuo
        if elitismo:
            mejor_individuo = poblacion[np.argmax(fitness_poblacion)]
            nueva_poblacion.append(mejor_individuo)

        # Bucle para generar el resto de la población
        while len(nueva_poblacion) < tam_poblacion:
            # 4. Selección
            if tipo_cromosoma == 'binario':
                padre1 = seleccion_torneo(poblacion, fitness_poblacion)
                padre2 = seleccion_torneo(poblacion, fitness_poblacion)
            else: # 'real'
                padre1 = seleccion_ruleta(poblacion, fitness_poblacion)
                padre2 = seleccion_ruleta(poblacion, fitness_poblacion)
            
            # 5. Cruce
            if tipo_cromosoma == 'binario':
                hijo1, hijo2 = cruce_un_punto(padre1, padre2)
            else: # 'real'
                hijo1, hijo2 = cruce_aritmetico(padre1, padre2)
            
            # 6. Mutación
            if tipo_cromosoma == 'binario':
                hijo1 = mutacion_bit_flip(hijo1, prob_mutacion)
                hijo2 = mutacion_bit_flip(hijo2, prob_mutacion)
            else: # 'real'
                hijo1 = mutacion_gaussiana(hijo1, prob_mutacion)
                hijo2 = mutacion_gaussiana(hijo2, prob_mutacion)
                
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(hijo2)
        
        poblacion = nueva_poblacion

    # Evaluación final para encontrar el mejor absoluto
    fitness_final = [fitness_func(ind, **kwargs_fitness) for ind in poblacion]
    mejor_indice_final = np.argmax(fitness_final)
    
    return {
        'mejor_solucion': poblacion[mejor_indice_final],
        'mejor_fitness': fitness_final[mejor_indice_final],
        'historial_mejor_fitness': historial_mejor_fitness,
        'historial_fitness_promedio': historial_fitness_promedio,
        'historial_diversidad': historial_diversidad,
    }