# Film_Junky_Union_Project

Este proyecto forma parte del programa de aprendizaje de ciencia de datos y tiene como objetivo aplicar **técnicas de machine learning al análisis de sentimientos en textos**.  

En colaboración con **Film Junky Union**, una comunidad de aficionados al cine clásico, se busca construir un sistema capaz de **detectar automáticamente críticas negativas** en reseñas de películas.  

---

## Objetivo  

Entrenar un modelo de clasificación de reseñas de películas del conjunto de datos **IMDB reviews**, alcanzando un valor de **F1 ≥ 0.85** para distinguir entre reseñas positivas y negativas.  

---

## Descripción del dataset  

El conjunto de datos proviene de **Maas et al. (2011), "Learning Word Vectors for Sentiment Analysis"**, ACL 2011.  

Archivo: `[imdb_reviews.tsv](https://practicum-content.s3.us-west-1.amazonaws.com/datasets/imdb_reviews.tsv?etag=bbd4a8dc10e6cf1280c42d7350105c41)`  

Columnas principales:  

- `review`: texto de la reseña.  
- `pos`: etiqueta objetivo (`0` = negativo, `1` = positivo).  
- `ds_part`: parte del dataset (`train` / `test`).  

---

## Instrucciones del proyecto  

1. **Carga de datos**  
   - Lectura del dataset y exploración inicial.  

2. **Análisis exploratorio de datos (EDA)**  
   - Identificación de distribución de clases y posibles desequilibrios.  
   - Visualización de ejemplos y métricas básicas.  

3. **Preprocesamiento de texto**  
   - Limpieza y normalización.  
   - Tokenización y eliminación de stopwords.  
   - Representación vectorial (Bag of Words, TF-IDF, o embeddings).  

4. **Modelado**  
   - Entrenamiento de al menos tres modelos diferentes, por ejemplo:  
     - Regresión Logística.  
     - Potenciación del gradiente (Gradient Boosting).  
     - Naive Bayes o SVM.  
   - Opcional: experimentos con **BERT embeddings** en una muestra reducida.  

5. **Evaluación**  
   - Medición con métricas de precisión, recall y F1-score.  
   - El modelo debe lograr **F1 ≥ 0.85** en el conjunto de prueba.  

6. **Pruebas personalizadas**  
   - Clasificación de reseñas escritas manualmente.  
   - Comparación de predicciones entre modelos.  

7. **Conclusiones**  
   - Análisis de resultados.  
   - Diferencias en desempeño de los modelos.  
   - Posibles mejoras y trabajo futuro.  

---

## Tecnologías utilizadas  

- [Python 3](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [NLTK](https://www.nltk.org/) o [spaCy](https://spacy.io/) (para procesamiento de texto).  
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) para visualización.  
- Opcional: [Transformers](https://huggingface.co/transformers/) para BERT.  

