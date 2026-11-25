# Predicción de precios de viviendas

**LI22 — Proyecto ML + Exposición**

Jhael Elisa Guevara Silva

Universidad del Valle · 2026

---

## Problema y objetivos

**Escenario 1:** predicción de precios de viviendas en Melbourne.

- **Objetivo:** aplicar tres enfoques de Machine Learning al mismo problema.
- **Supervisado:** estimar el precio de una vivienda.
- **No supervisado:** segmentar viviendas por características.
- **Refuerzo:** aprender la mejor estrategia de precio de publicación.

---

## Dataset y justificación

**Melbourne Housing Market — Kaggle**

| Requisito | Dataset |
| --- | --- |
| Registros | 34.857 |
| Columnas | 21 |
| Categóricas | 8 |
| Numéricas | 13 |
| Variable objetivo | `Price` |

Cumple los mínimos (5.000 registros, 13 columnas, 3 categóricas, 10 numéricas) y
permite los tres enfoques.

---

## Análisis exploratorio: hallazgos

- 7.610 precios nulos.
- `BuildingArea` y `YearBuilt` con más del 50 % de nulos.
- Precio asimétrico con valores atípicos.
- Outlier extremo en `BuildingArea` (44.515 m²).
- Variables más prometedoras: `Rooms`, `BuildingArea` y ubicación.

---

## Preprocesamiento

- Eliminación de duplicados y de filas sin precio.
- Imputación de nulos numéricos con la mediana.
- Corrección de valores inválidos y outliers.
- Creación de `Antiguedad`.
- One-Hot Encoding de `Regionname`.
- Estandarización con `StandardScaler`.

Resultado: 27.247 viviendas listas para modelar.

---

## Aprendizaje supervisado: regresión lineal

**Modelo:** `y = b0 + b1*x1 + ... + bp*xp`

**Función objetivo:** minimizar `sum((yi - yi_hat)^2)`.

- Predictoras: `Rooms`, `BuildingArea`, `Antiguedad`, región.
- División: 80 % entrenamiento / 20 % prueba.
- Evaluación: MAE, MSE, RMSE y R².
- Análisis de coeficientes y residuos.

---

## Supervisado: resultados

| Métrica | Valor |
| --- | --- |
| MAE | $322.028 |
| RMSE | $490.551 |
| R² | 0,437 |

- Explica ~44 % de la varianza del precio.
- Los residuos dispersos indican variables omitidas.
- Útil como línea base interpretable.

---

## No supervisado: K-Means

**Objetivo:** minimizar la inercia `sum ||x - mk||²`.

- Variables: `Rooms`, `BuildingArea`, `Distance`, `Antiguedad`.
- Selección de `k` con codo + silhouette.
- Se elige `k = 3`: económico, mediano y premium.

`s(i) = (b(i) - a(i)) / max(a(i), b(i))`

---

## No supervisado: resultados

| Segmento | Viviendas | Perfil |
| --- | --- | --- |
| 0 | 15.698 | Económico |
| 1 | 8.867 | Mediano |
| 2 | 2.682 | Premium |

**Silhouette = 0,308**: separación moderada, grupos interpretables.

---

## Refuerzo: entorno Multi-Armed Bandits

| Elemento | Definición |
| --- | --- |
| Estado | Bandit sin estado (comprador potencial) |
| Acciones | Precio Alto, Medio, Bajo |
| Recompensa | Precio si acepta; 0 si rechaza |
| Terminación | 1.000 episodios |
| Parámetros | epsilon = 0,1 |

Política **epsilon-greedy**: explorar vs. explotar.

---

## Refuerzo: resultados

- Ingreso esperado: Alto $100.000 · Medio $480.000 · Bajo $425.000.
- El agente aprende a preferir la estrategia **Medio**.
- La recompensa acumulada crece de forma sostenida.
- Alto y Bajo se exploran cada vez menos.

---

## Comparación de los tres enfoques

| Enfoque | Algoritmo | Resultado |
| --- | --- | --- |
| Supervisado | Regresión Lineal | R² = 0,437 |
| No supervisado | K-Means (k=3) | Silhouette = 0,308 |
| Refuerzo | Multi-Armed Bandits | Mejor acción: Medio |

Son complementarios: estimar, segmentar y decidir.

---

## Conclusiones y limitaciones

**Conclusiones**

- El dataset cumple todos los requisitos.
- La regresión ofrece una línea base interpretable.
- K-Means produce segmentos útiles.
- El agente aprende la estrategia de mayor ingreso esperado.

**Limitaciones**

- Nulos y outliers en variables clave.
- Variables omitidas.
- Entorno de refuerzo simulado.

---

## Mejoras futuras

- Imputación más robusta.
- Modelos no lineales (Random Forest, Gradient Boosting).
- Clustering jerárquico o DBSCAN como contraste.
- Bandit contextual por vivienda.

---

## Referencias

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning*. Springer.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
