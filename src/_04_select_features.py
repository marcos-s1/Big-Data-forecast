import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import warnings
from tqdm.auto import tqdm
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, abs as spark_abs
import datetime

# Define a opção para mostrar todas as colunas
pd.set_option('display.max_columns', None)

# Se quiser ver todas as linhas também
pd.set_option('display.max_rows', 100)

# Ignora avisos para uma saída mais limpa
warnings.filterwarnings('ignore')

# --- Funções Auxiliares para o Pandas ---
def calculate_wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula o Weighted Mean Absolute Percentage Error (WMAPE) para DataFrames Pandas."""
    if y_true.sum() == 0:
        return float('inf')
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def run_feature_selection(
  df: DataFrame, 
  reference_date_for_validation: 
  int, catboost_params: dict, 
  label_col: str, 
  id_cols: list, 
  ignore_features: list, 
  categorical_features: list, 
  sample_fraction: float):
    """
    Realiza a seleção de features em um pipeline híbrido (Spark -> Pandas).

    Args:
        df_spark (DataFrame): O DataFrame de entrada.
        reference_date_for_validation (int): data de referencia que será usada na validação out-of-time.
        catboost_params (dict): Dicionário de parâmetros do CatBoost.
        label_col (str): Nome da coluna target.
        id_cols (list): Lista de colunas de identificação.
        ignore_features (list): Lista de features a serem ignoradas na modelagem.
        categorical_features (list): Lista de features categóricas.

    Returns:
        list: Lista final de features selecionadas.
    """
    print("--- 1. Preparando os dados para a seleção de features ---")

    # --- Divisão Out-of-Time no Spark ---
  
    df_train = df[df["reference_date"] <= pd.to_datetime(df['reference_date']).dt.date]
    df_val = df[df["reference_date"] > pd.to_datetime(df['reference_date']).dt.date]

    # Define a lista completa de features candidatas
    all_features = [c for c in df.columns if c not in [label_col] + id_cols + ignore_features]

    # Preparando dados Pandas para o CatBoost
    X_train = df_train[all_features]
    y_train = df_train[label_col]
    X_val = df_val[all_features]
    y_val = df_val[label_col]

    # Convertendo as features categóricas para o tipo 'category'
    categorical_features_in_df = [c for c in all_features if c in categorical_features]
    for col_name in categorical_features_in_df:
        X_train[col_name] = X_train[col_name].astype(str)
        X_val[col_name] = X_val[col_name].astype(str)

    print(f"Conjunto de treino (amostra): {len(X_train)} linhas")
    print(f"Conjunto de validação (completo): {len(X_val)} linhas")

    # --- 2. Triagem Rápida por Feature Importance ---
    print("\n--- 2. Seleção Rápida por Feature Importance ---")
    model_quick = CatBoostRegressor(**catboost_params)
    model_quick.fit(X_train, y_train, cat_features=categorical_features_in_df, verbose=0)

    feature_importance_map = dict(zip(X_train.columns, model_quick.get_feature_importance()))

    # Filtra features com importância baixa
    initial_selected_features = [
        feat for feat, imp in feature_importance_map.items() if imp > 0.001
    ]
    initial_selected_features.sort(key=lambda x: feature_importance_map[x], reverse=False)

    print(f"Features após a triagem inicial: {len(initial_selected_features)} de {len(all_features)}")

    if not initial_selected_features:
        print("Nenhuma feature importante encontrada. Abortando a seleção.")
        return []

    # --- 3. Backward Selection com Validação ---
    print("\n--- 3. Iniciando Backward Selection com Validação ---")

    current_features = list(initial_selected_features)
    removed_features = []
    best_wmape_val = float('inf')

    with tqdm(total=len(initial_selected_features), desc="Backward Selection") as pbar:
        while len(current_features) > 1:
            wmape_results = []

            pbar_inner = tqdm(current_features, desc=f"Testando {len(current_features)} features", leave=False)
            for feature_to_remove in pbar_inner:
                temp_features = [f for f in current_features if f != feature_to_remove]

                model_back = CatBoostRegressor(**catboost_params)
                model_back.fit(X_train[temp_features], y_train, eval_set=(X_val[temp_features], y_val), cat_features=list(set(categorical_features_in_df) & set(temp_features)))

                y_pred_train = model_back.predict(X_train[temp_features])
                y_pred_val = model_back.predict(X_val[temp_features])

                wmape_train = calculate_wmape(y_train, y_pred_train)
                wmape_val = calculate_wmape(y_val, y_pred_val)

                # if abs(wmape_val/wmape_train - 1) > 0.10:
                #     pbar.write(f"Aviso: Overfitting detectado ao remover '{feature_to_remove}'. WMAPE de treino vs. validação: {wmape_train:.4f} vs {wmape_val:.4f}.")
                #     continue

                wmape_results.append({'feature_to_remove': feature_to_remove, 'wmape_val': wmape_val, 'diff_abs': abs(wmape_val - wmape_train)})

            pbar_inner.close()

            if not wmape_results:
                pbar.write("\nNenhuma feature atendeu aos critérios. Parando a eliminação.")
                break

            wmape_results_df = pd.DataFrame(wmape_results)
            wmape_results_df = wmape_results_df.sort_values(by=['wmape_val', 'diff_abs'])

            best_removal_row = wmape_results_df.iloc[0]
            feature_to_remove = best_removal_row['feature_to_remove']
            wmape_after_removal = best_removal_row['wmape_val']

            if wmape_after_removal < best_wmape_val:
                best_wmape_val = wmape_after_removal
                current_features.remove(feature_to_remove)
                pbar.write(f"-> Removida: '{feature_to_remove}'. Novo melhor WMAPE: {best_wmape_val:.4f}")
            else:
                pbar.write("\nO WMAPE não melhorou mais. Parando a eliminação.")
                break

            pbar.update(1)

    final_selected_features = current_features

    print("\n--- Processo de Seleção Concluído ---")
    print(f"Conjunto final de features: {final_selected_features}")
    return final_selected_features