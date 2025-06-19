# **CEIA - FIUBA**

>**Facultad de Ingeniería (Universidad de Buenos Aires)**

>**Especialización en Inteligencia Artificial**
---
# **Procesamiento de Lenguaje Natural I**

---

### **Alumno**
>**Nicolas Pinzon Aparicio**  
>**(a1820)**  
>**npinzonaparicio@gmail.com**

---

## **Contenido del repositorio**

Este repositorio contiene los notebooks desarrollados como parte de la materia **Procesamiento de Lenguaje Natural I**, organizados en cuatro desafíos principales que cubren desde técnicas fundamentales hasta modelos avanzados de deep learning.

---

## **Desafíos**

| Notebook | Temática Principal | Descripción |
|----------|-------------------|-------------|
| **Desafio_1.ipynb** | **Vectorización de Texto y Clasificación Naïve Bayes** | Análisis del dataset 20 Newsgroups mediante vectorización TF-IDF, estudio de similaridad coseno entre documentos, y entrenamiento de clasificadores Naïve Bayes (Multinomial y Complement) con optimización de hiperparámetros. Incluye análisis de similaridad entre palabras usando matrices término-documento. |
| **2c - Custom embedding con Gensim.ipynb** | **Word Embeddings Personalizados con Gensim** | Creación de embeddings customizados usando Word2Vec sobre un corpus musical de múltiples artistas (Beatles, Bob Dylan, Nirvana, Radiohead, Drake). Exploración de relaciones semánticas, visualización 2D/3D con t-SNE y PCA, y análisis de clusters temáticos. |
| **3_modelo_lenguaje_word.ipynb** | **Modelo de Lenguaje con Tokenización por Palabras** | Implementación de un modelo de lenguaje usando arquitecturas RNN (SimpleRNN, LSTM, GRU) entrenado en letras de canciones. Incluye generación de texto con estrategias Greedy Search, Beam Search y muestreo por temperatura. Monitoreo con callback de perplejidad. |
| **6- bot_qa.ipynb** | **Bot de Pregunta-Respuesta (QA Bot)** | Desarrollo de un chatbot usando arquitectura Encoder-Decoder con LSTM entrenado en el dataset ConvAI2. Implementación de modelos de inferencia separados y evaluación interactiva del bot con múltiples estrategias de generación de respuestas. |

---

## **⚙️ Requisitos técnicos**

### **Dependencias principales**
```bash
pip install tensorflow>=2.19.0
pip install scikit-learn>=1.6.0
pip install gensim>=4.3.0
pip install pandas numpy matplotlib seaborn
pip install plotly gdown nltk
```

### **Entorno recomendado**
- Python 3.9+
- Jupyter Notebook o Google Colab
- Mínimo 8GB RAM para entrenamiento de modelos

---

## **🚀 Características destacadas**

### **Desafío 1: Análisis Textual Avanzado**
- ✅ Vectorización TF-IDF con optimización de hiperparámetros
- ✅ Comparación de modelos MultinomialNB vs ComplementNB
- ✅ Análisis de similaridad documento-documento y palabra-palabra
- ✅ F1-score macro de 0.7078 en el mejor modelo

### **Desafío 2: Embeddings Musicales**
- ✅ Corpus rico de 16,160 líneas de 5 artistas diversos
- ✅ Word2Vec entrenado por 50 épocas con vocabulario de 2,947 palabras
- ✅ Visualizaciones interactivas con Plotly
- ✅ Análisis semántico por categorías (emociones, música, tiempo, personas)

### **Desafío 3: Generación de Texto Musical**
- ✅ Modelo LSTM optimizado con 1.37M parámetros
- ✅ Corpus de 10,710 líneas con vocabulario de 6,259 palabras
- ✅ Múltiples estrategias de generación (Greedy, Beam Search, Temperatura)
- ✅ Callback personalizado para monitoreo de perplejidad

### **Desafío 4: Chatbot Conversacional**
- ✅ Arquitectura Encoder-Decoder con embeddings compartidos
- ✅ Dataset ConvAI2 con 8,870 pares pregunta-respuesta
- ✅ Modelos de inferencia optimizados para deployment
- ✅ Evaluación interactiva con chat en tiempo real

---

## **📊 Resultados obtenidos**

| Desafío | Métrica Principal | Resultado | Observaciones |
|---------|------------------|-----------|---------------|
| 1 | F1-score macro | **0.7078** | ComplementNB superó a MultinomialNB |
| 2 | Coherencia semántica | **Alta** | Clusters temáticos bien definidos |
| 3 | Perplejidad final | **298.41** | Generación coherente a nivel local |
| 4 | Accuracy validación | **0.2654** | Bot funcional para conversaciones básicas |

---

## **🎯 Uso de los notebooks**

### **Para ejecutar individualmente:**
```python
# Desafío 1 - Clasificación
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Desafío 2 - Word2Vec
from gensim.models import Word2Vec
model.wv.most_similar('love', topn=5)

# Desafío 3 - Generación de texto
generate_text_with_temperature(model, tokenizer, "love is", max_length=10, temperature=1.0)

# Desafío 4 - QA Bot
response = generate_response("How are you?")
```

### **Para chat interactivo (Desafío 4):**
```python
chat_with_bot()  # Inicia sesión de chat
```

---

## **📝 Estructura del proyecto**

```
├── Desafio_1.ipynb                                   # Vectorización y Naïve Bayes
├── 2c - Custom embedding con Gensim.ipynb            # Word Embeddings con Gensim  
├── 3_modelo_lenguaje_word.ipynb                      # Modelo de lenguaje RNN
├── 6- bot_qa.ipynb                                   # QA Bot Seq2Seq

```

---

## **🔬 Metodología aplicada**

Cada desafío sigue una metodología rigurosa:

1. **Análisis exploratorio** detallado de los datos
2. **Preprocesamiento** específico para cada tarea
3. **Experimentación** con múltiples arquitecturas/parámetros
4. **Evaluación cuantitativa** con métricas apropiadas
5. **Análisis cualitativo** de resultados generados
6. **Documentación completa** de conclusiones y limitaciones

---

## **💡 Contribuciones técnicas**

- **Callbacks personalizados** para monitoreo de perplejidad
- **Visualizaciones interactivas** con Plotly para embeddings
- **Estrategias avanzadas** de generación de texto (Beam Search estocástico)
- **Arquitecturas optimizadas** para recursos limitados
- **Evaluación integral** combinando métricas automáticas y análisis humano


---

*Desarrollado como parte del posgrado en Inteligencia Artificial - FIUBA 2025*