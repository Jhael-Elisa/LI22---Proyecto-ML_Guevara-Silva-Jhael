"""Configuración común para todos los notebooks del proyecto.

Uso: al inicio de cada notebook ejecutar la celda:

    %run setup.py

Así las librerías, rutas y funciones quedan disponibles como variables
globales sin repetir código en cada cuaderno.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Rutas de datos (relativas a la carpeta notebooks/)
RAW_PATH = "../data/raw/Melbourne_housing_FULL.csv"
PROCESSED_PATH = "../data/processed/Melbourne_housing_FULL_PROCESADO.csv"


def load_raw():
    """Carga el dataset original de Kaggle."""
    return pd.read_csv(RAW_PATH)


def load_processed():
    """Carga el dataset procesado por 02_preprocessing.ipynb."""
    return pd.read_csv(PROCESSED_PATH)


def save_processed(df):
    """Guarda el dataset procesado para los notebooks siguientes."""
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Dataset procesado guardado en {PROCESSED_PATH}: {df.shape}")
