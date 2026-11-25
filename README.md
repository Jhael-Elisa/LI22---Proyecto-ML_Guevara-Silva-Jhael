# LI22 - Proyecto Machine Learning: Predicción de Precios de Viviendas

**Estudiante:** Jhael Elisa Guevara Silva  
**Materia:** LI22 - Proyecto ML + Exposición  
**Escenario Asignado:** N.º 1 - Predicción de precios de viviendas  
**Dataset:** Melbourne Housing Market (Fuente: Kaggle)

## 📌 Descripción del Proyecto
Este repositorio contiene la implementación de tres enfoques de Machine Learning aplicados al mercado inmobiliario de Melbourne, Australia. El objetivo es predecir el precio de las viviendas, segmentar las propiedades según sus características y simular estrategias de precios de publicación.

## 📂 Estructura del Repositorio
- `data/raw/`: Contiene el dataset original descargado de Kaggle (`Melbourne_housing_FULL.csv`).
- `notebooks/`: Cuadernos de Jupyter con el desarrollo paso a paso.
  - `01_EDA.ipynb`: Análisis Exploratorio de Datos.
  - `02_preprocessing.ipynb`: Limpieza y preparación de datos.
  - `03_supervised_model.ipynb`: Modelo de Regresión Lineal.
  - `04_unsupervised_model.ipynb`: Segmentación con K-Means.
  - `05_reinforcement_learning.ipynb`: Simulación con Multi-Armed Bandits.
  - `06_model_comparison.ipynb`: Conclusiones y comparación de resultados.
- `docs/`: Informe final en PDF y presentación de la exposición.

## 🛠️ Tecnologías y Librerías Utilizadas
- Python 3.13
- Pandas, NumPy (Manipulación de datos)
- Scikit-Learn (Modelos de ML)
- Matplotlib, Seaborn (Visualizaciones)

## 🚀 Cómo ejecutar
1. Clona este repositorio.
2. Abre los cuadernos de la carpeta `notebooks/` en Google Colab o Jupyter Notebook.
3. Asegúrate de subir el archivo `data/raw/Melbourne_housing_FULL.csv` antes de ejecutar los códigos.

## 📊 Resumen de los 3 Enfoques
1. **Aprendizaje Supervisado (Regresión Lineal):** Predicción del precio usando variables como habitaciones, superficie, antigüedad y ubicación. Evaluado con MAE, MSE, RMSE y R².
2. **Aprendizaje No Supervisado (K-Means):** Agrupación de viviendas en 3 segmentos (económicas, medianas, premium) mediante el Método del Codo.
3. **Aprendizaje por Refuerzo (Multi-Armed Bandits):** Simulación de un agente que aprende a elegir la mejor estrategia de publicación (precio alto, medio, bajo) interactuando con 1000 compradores simulados.

---
*Proyecto desarrollado para la Universidad del Valle - Bolivia.*