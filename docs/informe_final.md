# Predicción de precios de viviendas

## Informe final — LI22 Proyecto ML + Exposición

**Universidad del Valle**

**Estudiante:** Jhael Elisa Guevara Silva

**Escenario asignado:** N.º 1 — Predicción de precios de viviendas

**Dataset:** Melbourne Housing Market (Kaggle)

**Fecha:** 2026

---

## 1. Introducción

El mercado inmobiliario es un problema de alto interés económico y social. Estimar
el precio de una vivienda permite orientar compradores, vendedores e inmobiliarias,
mientras que segmentar el mercado facilita diseñar estrategias comerciales y elegir
un precio de publicación adecuado.

Este proyecto aplica tres enfoques complementarios de Machine Learning sobre el
dataset *Melbourne Housing Market*:

1. **Regresión lineal** para predecir el precio de una vivienda.
2. **K-Means** para segmentar viviendas según sus características.
3. **Multi-Armed Bandits** para simular y aprender estrategias de precio de publicación.

**Objetivo general:** aplicar y comparar los tres enfoques sobre un mismo escenario,
justificando los datos, los modelos, los parámetros y la interpretación de resultados.

**Objetivos específicos:**

- Verificar que el dataset cumple los requisitos mínimos de la materia.
- Realizar un análisis exploratorio y un preprocesamiento justificado.
- Entrenar y evaluar un modelo supervisado con MAE, MSE, RMSE y R².
- Determinar un número apropiado de grupos con K-Means y perfilarlos.
- Construir un entorno de refuerzo con estados, acciones, recompensas y episodios.
- Comparar los resultados e identificar limitaciones y mejoras.

---

## 2. Descripción del dataset

**Fuente:** Kaggle — Melbourne Housing Market.
<https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market>

**Tamaño:** 34.857 registros y 21 columnas.

**Tipos de datos:** 8 variables categóricas (`Suburb`, `Address`, `Type`, `Method`,
`SellerG`, `Date`, `CouncilArea`, `Regionname`) y 13 numéricas (`Rooms`, `Price`,
`Distance`, `Postcode`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`,
`YearBuilt`, `Lattitude`, `Longtitude`, `Propertycount`).

**Variable objetivo:** `Price`, precio de venta en dólares australianos.

### 2.1 Cumplimiento de requisitos

| Requisito | Valor en el dataset | Cumple |
| --- | --- | --- |
| Mínimo 5.000 registros | 34.857 filas | Sí |
| Mínimo 13 columnas | 21 columnas | Sí |
| Al menos 3 categóricas | 8 variables | Sí |
| Al menos 10 numéricas | 13 variables | Sí |
| Requiere preprocesamiento | Nulos, duplicados, codificación, escalado | Sí |
| Fuente identificable | Kaggle | Sí |

### 2.2 Justificación de la selección

El dataset permite abordar los tres enfoques porque contiene una variable objetivo
continua (regresión), varias características físicas y de ubicación (clustering) y
un contexto comercial donde distintos precios de publicación generan distintas
probabilidades de venta (refuerzo). Además, es público, trazable y de tamaño
suficiente.

### 2.3 Limitaciones

- `BuildingArea` y `YearBuilt` superan el 50 % de valores nulos.
- `Price` presenta asimetría positiva y valores atípicos.
- Algunas variables categóricas tienen alta cardinalidad (`Suburb`, `Address`).
- El dataset corresponde a 2016–2017 y no refleja el mercado actual.

---

## 3. Análisis exploratorio de datos (EDA)

El notebook `01_EDA.ipynb` desarrolla:

- Verificación automática de los requisitos mínimos del dataset.
- Conteo y porcentaje de valores nulos por columna.
- Detección de filas duplicadas.
- Estadísticas descriptivas (`describe`).
- Distribución del precio y de su transformación logarítmica.
- Frecuencias de `Type` y `Regionname`.
- Matriz de correlación entre variables numéricas.
- Detección de valores atípicos mediante rango intercuartílico.

**Hallazgos principales:**

1. Existen 7.610 precios nulos que deben eliminarse.
2. `BuildingArea` (21.115 nulos) y `YearBuilt` (19.306 nulos) requieren imputación.
3. El precio es asimétrico: la media supera la mediana por los inmuebles de lujo.
4. `Rooms`, `BuildingArea` y la ubicación son las variables más prometedoras.
5. Se detectan outliers extremos en `BuildingArea` (hasta 44.515 m²), corregidos
   en el preprocesamiento.

---

## 4. Preprocesamiento

El notebook `02_preprocessing.ipynb` aplica las siguientes decisiones:

| Tarea | Decisión | Justificación |
| --- | --- | --- |
| Duplicados | Eliminación | Evitar registros repetidos |
| Nulos en `Price` | Eliminación de filas | Es la variable objetivo |
| Nulos en `BuildingArea`, `YearBuilt`, `Distance` | Mediana | Robusta frente a outliers |
| Valores inválidos | `BuildingArea <= 0` y `YearBuilt` fuera de rango a nulo | Corregir errores de captura |
| Outliers de `BuildingArea` | Acotar al percentil 99 | Evitar distorsión en modelos |
| Fecha | Conversión a fecha y cálculo de `Antiguedad` | Usar la edad de la vivienda |
| `Regionname` | One-Hot Encoding (`drop_first=True`) | Convertir ubicación a numérico |
| Escalado | `StandardScaler` | Necesario para regresión y K-Means |

Tras el preprocesamiento quedan 27.247 viviendas y 28 columnas, sin nulos en las
variables predictoras. El resultado se guarda en
`data/processed/Melbourne_housing_FULL_PROCESADO.csv`.

---

## 5. Fundamentos teóricos y base matemática

### 5.1 Regresión lineal

Modelo:

`y = b0 + b1*x1 + b2*x2 + ... + bp*xp + e`

Función objetivo (mínimos cuadrados ordinarios):

`J(b) = sum( (yi - yi_hat)^2 )`

Métricas de evaluación:

- `MAE = (1/n) * sum( |yi - yi_hat| )`
- `MSE = (1/n) * sum( (yi - yi_hat)^2 )`
- `RMSE = sqrt(MSE)`
- `R² = 1 - SS_res / SS_tot`

Supuestos: linealidad, independencia de errores, homocedasticidad y normalidad
de los residuos.

### 5.2 K-Means

Objetivo: minimizar la inercia.

`J = sum(k=1..K) sum(x en Ck) ||x - mk||²`

El algoritmo alterna asignación de puntos al centroide más cercano y recálculo de
centroides como la media del grupo, hasta converger.

Calidad de los grupos con silhouette:

`s(i) = (b(i) - a(i)) / max(a(i), b(i))`

donde `a(i)` es la distancia media al propio grupo y `b(i)` al grupo vecino.
El rango es [-1, 1] y valores mayores indican mejor separación.

### 5.3 Multi-Armed Bandits

Cada acción `a` tiene una recompensa esperada desconocida `mu(a)`. El agente estima:

`Q(a) = suma de recompensas de a / veces que eligió a`

La política **epsilon-greedy** elige la mejor acción con probabilidad `1 - epsilon`
y explora al azar con probabilidad `epsilon`. El objetivo es maximizar la
recompensa acumulada y reducir el arrepentimiento (*regret*) frente a la mejor acción.

---

## 6. Implementación

**Lenguaje y librerías:** Python 3.13+, Pandas, NumPy, Scikit-Learn, Matplotlib,
Seaborn y JupyterLab. Las versiones mínimas están en `requirements.txt`.

Cada notebook ejecuta `%run setup.py`, que carga librerías, rutas y funciones
comunes (`load_raw`, `load_processed`, `save_processed`) como variables globales,
evitando duplicar código.

### 6.1 Modelo supervisado

- Variables: `Rooms`, `BuildingArea`, `Antiguedad` y dummies de `Regionname`.
- División: 80 % entrenamiento, 20 % prueba, `random_state=42`.
- Modelo: `LinearRegression` de Scikit-Learn.
- Análisis: coeficientes, intercepto, residuos y métricas.

### 6.2 Modelo no supervisado

- Variables: `Rooms`, `BuildingArea`, `Distance`, `Antiguedad`.
- Estandarización con `StandardScaler`.
- Rango evaluado: `k = 2` a `10`.
- Selección de `k` con método del codo y silhouette score.
- `k = 3` para obtener perfiles económico, mediano y premium.

### 6.3 Aprendizaje por refuerzo

| Elemento | Definición |
| --- | --- |
| Estado | Bandit sin estado: cada episodio representa un comprador potencial |
| Acciones | Precio Alto ($1.000.000), Medio ($800.000), Bajo ($500.000) |
| Recompensa | Precio de venta si el comprador acepta; 0 si rechaza |
| Terminación | Un episodio por comprador; 1.000 episodios en total |
| Parámetros | `epsilon = 0.1`, 3 acciones, semilla 42 |

Probabilidades simuladas de aceptación: Alto 0,10; Medio 0,60; Bajo 0,85. El
ingreso esperado resultante es $100.000, $480.000 y $425.000 respectivamente, por
lo que la estrategia óptima es **Medio**.

---

## 7. Entrenamiento y evaluación

### 7.1 Resultados del modelo supervisado

| Métrica | Valor |
| --- | --- |
| MAE | $322.028 |
| MSE | 240.640.000.000 aprox. |
| RMSE | $490.551 |
| R² | 0,437 |

El modelo explica aproximadamente el 44 % de la varianza del precio. Los
coeficientes muestran que `Rooms`, `BuildingArea` y algunas regiones tienen
efecto positivo, mientras que una mayor `Antiguedad` tiende a reducir el precio.
Los residuos presentan dispersión creciente, lo que sugiere heterocedasticidad y
variables omitidas.

### 7.2 Resultados del clustering

| Segmento | Viviendas | Perfil |
| --- | --- | --- |
| 0 | 15.698 | Menor superficie y precio: económico |
| 1 | 8.867 | Superficie y precio intermedios: mediano |
| 2 | 2.682 | Mayor superficie y precio: premium |

Silhouette con `k = 3`: **0,308**, lo que indica una separación moderada pero
útil para perfilar el mercado.

### 7.3 Resultados del agente

El agente epsilon-greedy concentra sus elecciones en la estrategia **Medio**,
que maximiza el ingreso esperado. La recompensa acumulada crece de forma sostenida
y las estrategias Alto y Bajo se exploran cada vez menos.

---

## 8. Análisis comparativo e interpretación

| Enfoque | Algoritmo | Pregunta | Resultado | Interpretación |
| --- | --- | --- | --- | --- |
| Supervisado | Regresión Lineal | ¿Cuánto vale una vivienda? | R² = 0,437 | Captura la tendencia general, pero omite factores cualitativos |
| No supervisado | K-Means (k=3) | ¿Qué perfiles existen? | Silhouette = 0,308 | Segmentos interpretables para estrategias diferenciadas |
| Refuerzo | Multi-Armed Bandits | ¿Qué precio publicar? | Mejor acción: Medio | Equilibra probabilidad de venta e ingreso |

Los tres enfoques son complementarios: la regresión estima el valor, el clustering
describe el mercado y el agente aprende una política de publicación. La comparación
es conceptual porque las tareas y métricas son distintas.

---

## 9. Conclusiones, limitaciones y mejoras

**Conclusiones:**

- El dataset cumple todos los requisitos y permite abordar los tres enfoques.
- La regresión lineal ofrece una línea base interpretable, con R² moderado.
- K-Means con `k = 3` produce segmentos coherentes con el mercado inmobiliario.
- El agente de refuerzo aprende a preferir la estrategia de mayor ingreso esperado.

**Limitaciones:**

- Alta proporción de nulos en `BuildingArea` y `YearBuilt`.
- Outliers y asimetría del precio.
- Variables omitidas: estado de conservación, calidad de construcción, submercado.
- Entorno de refuerzo simulado con probabilidades supuestas.

**Mejoras:**

- Imputación múltiple o modelos que manejen nulos nativamente.
- Modelos no lineales (Random Forest, Gradient Boosting) para comparar con la regresión.
- Clustering jerárquico o DBSCAN como contraste.
- Bandit contextual que use las características de cada vivienda.

---

## 10. Referencias

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3.ª ed.). O'Reilly Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2.ª ed.). Springer.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.). MIT Press.
