import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import warnings

# Ignora avisos do CatBoost para uma saída mais limpa
warnings.filterwarnings('ignore')

def wmape(y_true, y_pred):
    """Calcula o Weighted Mean Absolute Percentage Error (WMAPE)."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def run_feature_selection(df, target, identifiers, ignore_features, categorical_features, weeks_for_validation):
    """
    Executa a seleção de features usando backward elimination e WMAPE.
    
    Args:
        df (pd.DataFrame): DataFrame com todas as features e a variável-alvo.
        target (str): Nome da variável-alvo.
        identifiers (list): Lista de colunas de identificação.
        ignore_features (list): Lista de features a serem ignoradas na modelagem.
        categorical_features (list): Lista de features categóricas.
        weeks_for_validation (int): Número de semanas a serem usadas para validação.
        
    Returns:
        list: Lista com os nomes das features selecionadas.
    """
    print("--- 1. Preparando os dados para a seleção de features ---")
    
    # Identifica as variáveis que serão usadas como features
    all_features = [col for col in df.columns if col not in [target] + identifiers + ignore_features]
    
    # Converte colunas categóricas para o tipo 'category' para otimização do CatBoost
    for col_name in categorical_features:
        if col_name in df.columns:
            df[col_name] = df[col_name].astype('category')
        
    # --- 2. Divisão Out-of-Time para Treino e Validação ---
    print("--- 2. Realizando a divisão de dados Out-of-Time ---")
    cutoff_week = df['Semana'].max() - weeks_for_validation
    df_train_set = df[df['Semana'] <= cutoff_week].copy()
    df_val_set = df[df['Semana'] > cutoff_week].copy()

    X_train = df_train_set[all_features]
    y_train = df_train_set[target]
    X_val = df_val_set[all_features]
    y_val = df_val_set[target]
    
    print(f"Conjunto de treino: {len(X_train)} linhas")
    print(f"Conjunto de validação: {len(X_val)} linhas")

    # --- 3. Backward Selection (Eliminação Reversa) ---
    print("\n--- 3. Iniciando Backward Selection (Pode levar tempo) ---")

    current_features = list(all_features)
    removed_features = []
    best_wmape = 1e9 # Um valor alto para ser facilmente superado
    
    while len(current_features) > 1:
        wmape_scores = {}
        for feature_to_remove in current_features:
            test_features = [f for f in current_features if f != feature_to_remove]
            
            model = CatBoostRegressor(verbose=0, random_state=42)
            model.fit(X_train[test_features], y_train, cat_features=list(set(categorical_features) & set(test_features)))
            y_pred = model.predict(X_val[test_features])
            wmape_scores[feature_to_remove] = wmape(y_val, y_pred)
        
        best_removal = min(wmape_scores, key=wmape_scores.get)
        wmape_after_removal = wmape_scores[best_removal]

        if wmape_after_removal < best_wmape:
            best_wmape = wmape_after_removal
            removed_features.append(best_removal)
            current_features.remove(best_removal)
            print(f"-> Removida: '{best_removal}'. Novo melhor WMAPE: {best_wmape:.4f}")
        else:
            print("\nO WMAPE não melhorou mais. Parando a eliminação.")
            break
            
    final_features = current_features
    print("\n--- Backward Selection Finalizada ---")
    print(f"Conjunto de features selecionado: {final_features}")
    print(f"WMAPE alcançado: {best_wmape:.4f}")

    # --- 4. Repescagem de Variáveis ---
    print("\n--- 4. Iniciando Repescagem de Variáveis... ---")
    for feature_to_rescue in removed_features:
        temp_features = list(final_features)
        temp_features.append(feature_to_rescue)
        
        model = CatBoostRegressor(verbose=0, random_state=42)
        model.fit(X_train[temp_features], y_train, cat_features=list(set(categorical_features) & set(temp_features)))
        y_pred = model.predict(X_val[temp_features])
        wmape_rescued = wmape(y_val, y_pred)
        
        if wmape_rescued < best_wmape:
            best_wmape = wmape_rescued
            final_features.append(feature_to_rescue)
            print(f"-> Resgatada: '{feature_to_rescue}'. WMAPE melhorou para {wmape_rescued:.4f}")
        else:
            print(f"-> Não resgatada: '{feature_to_rescue}'. WMAPE não melhorou.")

    print("\n--- Processo de Seleção de Features Concluído ---")
    print(f"Conjunto final de features: {final_features}")
    return final_features

if __name__ == "__main__":
    print("--- Executando o script 03_select_features.py ---")
    input_path = "data/processed/base_com_features.parquet"
    
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado. Execute 02_feature_engineering.py primeiro.")
    else:
        df_com_features = pd.read_parquet(input_path)
        
        # --- CONFIGURAÇÃO DO MODELO ---
        TARGET = 'Vendas_Semanais'
        IDENTIFIERS = ['internal_store_id', 'internal_product_id']
        IGNORE_FEATURES = ['descricao', 'distributor_id'] # Exemplo: variáveis que você decide ignorar
        CATEGORICAL_FEATURES = [
            'Semana', 'week_of_month', 'distributor_id', 'categoria', 
            'subcategoria', 'marca', 'fabricante', 'tipos', 'label', 'descricao'
        ]
        WEEKS_FOR_VALIDATION = 4

        features_selecionadas = run_feature_selection(
            df_com_features, TARGET, IDENTIFIERS, IGNORE_FEATURES, CATEGORICAL_FEATURES, WEEKS_FOR_VALIDATION
        )
        
        output_path = "models/features_selecionadas.txt"
        with open(output_path, "w") as f:
            for item in features_selecionadas:
                f.write(f"{item}\n")
        print(f"\nLista de features selecionadas salva em '{output_path}'.")