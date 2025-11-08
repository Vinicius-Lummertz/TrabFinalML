import pandas as pd
import numpy as np

def to_string(X_input):
    """
    Converte a entrada (que pode ser um DataFrame ou ndarray) para string
    para garantir que o OneHotEncoder funcione.
    """
    # O FunctionTransformer pode passar um ndarray, então é melhor
    # convertê-lo para DataFrame para garantir o .astype(str)
    if isinstance(X_input, np.ndarray):
        X_df = pd.DataFrame(X_input)
    else:
        X_df = X_input.copy() # Usar .copy() para evitar SettingWithCopyWarning
        
    # Converte todas as colunas para string
    for col in X_df.columns:
        X_df[col] = X_df[col].astype(str)
    
    return X_df

"""Utility helpers required by the model unpickler."""

def to_string(value) -> str:
    """Convert value to string. Present so joblib unpickler can find the function by name."""
    return str(value)