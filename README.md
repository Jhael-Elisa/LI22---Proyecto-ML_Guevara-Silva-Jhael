# repositorio-TI26-Guevara

**Universidad del Valle — LI22 · Proyecto ML + Exposición**

**Estudiante:** Jhael Elisa Guevara Silva
**Escenario asignado:** N.º 1 — Predicción de precios de viviendas
**Dataset:** Melbourne Housing Market (Kaggle)
**Fecha:** 2026

---

## 1. Descripción del proyecto

El proyecto aplica los tres enfoques de Machine Learning exigidos por la materia
sobre el mercado inmobiliario de Melbourne, Australia:

1. **Aprendizaje supervisado:** regresión lineal para estimar el precio de una vivienda.
2. **Aprendizaje no supervisado:** K-Means para segmentar viviendas por sus características.
3. **Aprendizaje por refuerzo:** Multi-Armed Bandits para elegir la estrategia de precio de publicación.

El objetivo es predecir, segmentar y apoyar decisiones de publicación, no solo
ejecutar código: cada notebook justifica el dataset, preprocesa los datos,
explica el modelo, lo evalúa e interpreta los resultados.

---

## 2. Dataset y justificación

**Fuente:** Kaggle — *Melbourne Housing Market*
<https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market>

**Archivo:** `data/raw/Melbourne_housing_FULL.csv`

### 2.1 Cumplimiento de requisitos

| Requisito del documento | Valor en el dataset | Cumple |
| --- | --- | --- |
| Mínimo 5.000 registros | 34.857 filas | ✅ |
| Mínimo 13 columnas | 21 columnas | ✅ |
| Al menos 3 variables categóricas | 8: `Suburb`, `Address`, `Type`, `Method`, `SellerG`, `Date`, `CouncilArea`, `Regionname` | ✅ |
| Al menos 10 variables numéricas | 13: `Rooms`, `Price`, `Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`, `Propertycount` | ✅ |
| Requiere preprocesamiento | Nulos, duplicados, codificación y escalado | ✅ |
| Fuente identificable y variables descritas | Kaggle + tabla de variables | ✅ |
| Permite los tres enfoques | Precio continuo + características + estrategias de publicación | ✅ |

### 2.2 Justificación de la selección

- El **precio** es una variable objetivo continua, adecuada para regresión lineal.
- Las **características físicas y de ubicación** permiten segmentar viviendas con K-Means.
- Las **estrategias de precio de publicación** permiten construir un entorno de
  refuerzo donde un comprador simulado acepta o rechaza la oferta.
- El dataset es público, trazable y suficientemente grande para entrenar y evaluar.

### 2.3 Variables principales

| Variable | Tipo | Descripción |
| --- | --- | --- |
| `Suburb` | Categórica | Barrio de la vivienda |
| `Address` | Categórica | Dirección |
| `Rooms` | Numérica | Número de habitaciones |
| `Type` | Categórica | Tipo: casa (h), unidad (u), townhouse (t) |
| `Price` | Numérica | Precio de venta en AUD (variable objetivo) |
| `Method` | Categórica | Método de venta |
| `SellerG` | Categórica | Inmobiliaria |
| `Date` | Fecha | Fecha de venta |
| `Distance` | Numérica | Distancia al centro de Melbourne (km) |
| `Postcode` | Numérica | Código postal |
| `Bedroom2` | Numérica | Número de dormitorios |
| `Bathroom` | Numérica | Número de baños |
| `Car` | Numérica | Capacidad de estacionamiento |
| `Landsize` | Numérica | Superficie del terreno (m²) |
| `BuildingArea` | Numérica | Superficie construida (m²) |
| `YearBuilt` | Numérica | Año de construcción |
| `CouncilArea` | Categórica | Municipio |
| `Lattitude` / `Longtitude` | Numérica | Coordenadas |
| `Regionname` | Categórica | Región de Melbourne |
| `Propertycount` | Numérica | Viviendas en el barrio |

**Limitaciones:** `BuildingArea` y `YearBuilt` tienen más del 50 % de nulos y se
imputan con la mediana; el precio presenta asimetría y valores atípicos.

---

## 3. Estructura del repositorio

```
repositorio-TI26-Guevara/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/Melbourne_housing_FULL.csv
│   └── processed/Melbourne_housing_FULL_PROCESADO.csv
├── notebooks/
│   ├── setup.py                      # configuración común (%run setup.py)
│   ├── 01_EDA.ipynb                  # análisis exploratorio
│   ├── 02_preprocessing.ipynb        # limpieza, nulos, codificación, escalado
│   ├── 03_supervised_model.ipynb     # regresión lineal
│   ├── 04_unsupervised_model.ipynb   # K-Means + codo + silhouette
│   ├── 05_reinforcement_learning.ipynb  # Multi-Armed Bandits
│   └── 06_model_comparison.ipynb     # comparación e interpretación
└── docs/
    ├── informe_final.pdf
    └── presentacion.pdf
```

Cada notebook comienza con `%run setup.py`, que carga las librerías, rutas y
funciones comunes como variables globales, sin duplicar código.

---

## 4. Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Ejecutar los cuadernos **en orden** del 01 al 06. El notebook 02 genera el
archivo procesado que usan los notebooks 03 a 05.

---

## 5. Resultados principales

| Enfoque | Algoritmo | Resultado | Interpretación |
| --- | --- | --- | --- |
| Supervisado | Regresión Lineal | R² = 0,437 · MAE ≈ $322.028 · RMSE ≈ $490.551 | Explica ~44 % de la varianza del precio; útil como línea base, omite factores cualitativos. |
| No supervisado | K-Means (k = 3) | Silhouette = 0,308 · grupos de 15.698 / 8.867 / 2.682 viviendas | Separa perfiles económicos, medianos y premium. |
| Refuerzo | Multi-Armed Bandits (ε-greedy) | Mejor estrategia: precio **Medio** | Maximiza el ingreso esperado (probabilidad de venta × precio). |

---

## 6. Contenido de los notebooks

- **01_EDA:** requisitos del dataset, tipos de variables, nulos, duplicados,
  estadísticas descriptivas, distribuciones, categóricas y correlaciones.
- **02_preprocessing:** duplicados, nulos, valores inválidos, outliers,
  codificación de `Regionname`, variable `Antiguedad` y estandarización.
- **03_supervised_model:** fundamento teórico, coeficientes, residuos,
  MAE, MSE, RMSE y R².
- **04_unsupervised_model:** fundamento teórico, codo, silhouette, elección de
  `k`, tamaños de grupo y perfilamiento.
- **05_reinforcement_learning:** entorno (estado, acciones, recompensa,
  terminación, parámetros), entrenamiento y evolución de la recompensa.
- **06_model_comparison:** tabla comparativa de los tres enfoques, gráficos y
  conclusiones.

---

## 7. Referencias (APA)

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3.ª ed.). O'Reilly Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2.ª ed.). Springer.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.). MIT Press.

---

*Proyecto desarrollado para la Universidad del Valle — Bolivia.*
