# Sentiment-Estimator

Basic sentiment estimator (driven by numeric value) based on machine learning algorithms, it works well (and only) for the Spanish language.

Fue previamente entrenado con frases en español etiquetadas con el valor que determina la polaridad del sentimiento.

El proceso se basa en una extracción de caracteristicas (vectorización + selección) para luego apoyarse en modelos de aprendizaje automático (Red neuronal + Naive Bayes de Complemento).

El valor devuelto por la librería es un numero que indica la polaridad del sentimiento. Mientras más cercano a 2.0, el sentimiento será positivo, mientras más cercano a 0.0 el sentimiento será negativo. Los neutralidad se aproxima siempre a 1.0

## Uso

Mientras el estimador no esté publicado en un repositorio de paquetes, es necesario clonar este repositorio y realizar lo siguiente:

### Añadirlo al contexto:

`import sys`
`sys.path.append('<ruta_al_repo>/sentiment_estimator/lib')`

### Importar la librería:

`from sentiment_estimator import SentimentEstimator`

### Ejecutar:

`estimator = SentimentEstimator(context_path='<ruta_al_repo>/sentiment_estimator/data/')`
`estimator.predict('Yo creo que este producto es muy bueno') # Returns 1.89`
