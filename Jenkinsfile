// Jenkinsfile - pipeline de automatización
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Código clonado desde el repositorio.'
            }
        }
        
        stage('Instalar Dependencias') {
            steps {
                echo 'Instalando dependencias de Python...'
                sh 'pip install --user -r requirements.txt'
            }
        }
        
        stage('Pruebas Básicas') {
            steps {
                echo 'Ejecutando pruebas básicas...'
                sh '''
                python -c "
import pandas as pd
from src.genetic_operators import cruce_un_punto
print('Prueba de importación exitosa.')
df = pd.read_csv('data/sdss_sample.csv')
assert not df.empty, 'El dataset está vacío'
print(f'Dataset cargado con {len(df)} filas. Prueba exitosa.')
cromosoma1 = [0, 1, 0, 1]
cromosoma2 = [1, 0, 1, 0]
hijo1, hijo2 = cruce_un_punto(cromosoma1, cromosoma2)
assert len(hijo1) == len(cromosoma1)
print('Prueba de operador genético exitosa.')
"
                '''
            }
        }
        
        stage('Ejecutar Pipeline de AG') {
            steps {
                echo 'Ejecutando el script principal...'
                sh 'python main.py'
            }
        }
        
        stage('Archivar Artefactos') {
            steps {
                echo 'Archivando resultados...'
                archiveArtifacts artifacts: 'outputs/**/*', fingerprint: true
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline finalizado.'
        }
        success {
            echo '¡El pipeline se completó exitosamente!'
        }
        failure {
            echo 'El pipeline falló. Revisa los logs.'
        }
    }
}