from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, abs
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import numpy as np
from tqdm.auto import tqdm

def calculate_wmape(df: DataFrame, prediction_col: str, label_col: str) -> float:
    """
    Calcula o Weighted Mean Absolute Percentage Error (WMAPE).
    
    Args:
        df (DataFrame): DataFrame com as colunas de predição e label.
        prediction_col (str): Nome da coluna de predição.
        label_col (str): Nome da coluna de label (valor real).
        
    Returns:
        float: O valor do WMAPE.
    """
    total_sales = df.agg({label_col: "sum"}).collect()[0][0]
    if total_sales == 0:
        return float('inf')
    
    df_wmape = df.withColumn("abs_error", abs(col(label_col) - col(prediction_col)))
    
    total_abs_error = df_wmape.agg({"abs_error": "sum"}).collect()[0][0]
    
    return total_abs_error / total_sales

def select_and_validate_features(df: DataFrame, weeks_for_validation: int, gbt_params: dict, label_col: str, id_cols: list, feature_cols: list, sample_fraction: float):
    """
    Realiza a seleção de features em duas etapas.
    
    Args:
        df (DataFrame): O DataFrame com todas as features.
        weeks_for_validation (int): O número de semanas a ser usado para a validação out-of-time.
        gbt_params (dict): Dicionário de parâmetros do GBTRegressor.
        label_col (str): Nome da coluna target.
        id_cols (list): Lista de colunas de identificação para não serem usadas no modelo.
        feature_cols (list): Lista de todas as colunas de features candidatas.
        sample_fraction (float): Fração de amostragem a ser usada para o treino.
        
    Returns:
        list: Lista final de features selecionadas.
    """
    print("--- 1. Seleção Inicial de Features por Importância ---")
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_assembled = assembler.transform(df)

    max_week_rank = df.agg({"week_rank": "max"}).collect()[0][0]
    cutoff_week = max_week_rank - weeks_for_validation
    df_train = df_assembled.filter(col('week_rank') <= cutoff_week).cache()
    df_val = df_assembled.filter(col('week_rank') > cutoff_week).cache()

    # --- CORREÇÃO: Agora a lógica de amostragem é feita uma única vez ---
    if sample_fraction < 1.0:
        print(f"Gerando uma amostra de {sample_fraction*100}% dos dados de treino...")
        df_train_sample = df_train.sample(withReplacement=False, fraction=sample_fraction, seed=42)
    else:
        print("Usando o DataFrame de treino completo.")
        df_train_sample = df_train # Retorna o DataFrame original se a fração for 1.0

    print(f"Dividindo a base: Treino (até week_rank {cutoff_week}) e Validação (após week_rank {cutoff_week})")

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=label_col,
        maxIter=gbt_params.get('maxIter', 20),
        maxDepth=gbt_params.get('maxDepth', 5),
        stepSize=gbt_params.get('stepSize', 0.1),
        seed=gbt_params.get('seed', 42)
    )
    
    model = gbt.fit(df_train_sample)
    
    feature_importances = model.featureImportances
    
    feature_importance_map = {
        feature_cols[i]: importance 
        for i, importance in enumerate(feature_importances)
    }
    
    initial_selected_features = [
        feat for feat, imp in feature_importance_map.items() if imp > 0.001
    ]
    
    initial_selected_features.sort(key=lambda x: feature_importance_map[x], reverse=False)
    
    print(f"Features selecionadas após o filtro inicial: {len(initial_selected_features)} de {len(feature_cols)}")
    print(f"Features para o Backward Selection: {initial_selected_features}")
    
    if not initial_selected_features:
        print("Nenhuma feature importante encontrada. Abortando a seleção sequencial.")
        return []

    print("\n--- 2. Seleção Sequencial (Backward Selection) ---")
    
    best_features = initial_selected_features.copy()
    current_features = best_features.copy()
    
    assembler_base = VectorAssembler(inputCols=current_features, outputCol="features")
    df_train_base = assembler_base.transform(df_train_sample) # Usa a amostra para o treino base
    df_val_base = assembler_base.transform(df_val)
    
    gbt_base = GBTRegressor(
        featuresCol="features",
        labelCol=label_col,
        maxIter=gbt_params.get('maxIter', 20),
        maxDepth=gbt_params.get('maxDepth', 5),
        stepSize=gbt_params.get('stepSize', 0.1),
        seed=gbt_params.get('seed', 42)
    ).fit(df_train_base)

    predictions_base = gbt_base.transform(df_val_base)
    
    wmape_train_base = calculate_wmape(gbt_base.transform(df_train_base), 'prediction', label_col)
    wmape_val_base = calculate_wmape(predictions_base, 'prediction', label_col)
    
    best_overfitting_score = abs(wmape_val_base / wmape_train_base - 1)
    
    print(f"WMAPE Inicial (Val): {wmape_val_base:.4f}, Overfitting Score: {best_overfitting_score:.4f}")
    
    removed_features = []
    
    # Adicionando tqdm ao loop
    with tqdm(initial_selected_features, desc="Backward Selection") as pbar:
        for feat_to_remove in pbar:
            temp_features = [f for f in current_features if f != feat_to_remove]
            
            if not temp_features:
                pbar.write("Nenhuma feature para remover. Processo concluído.")
                break
            
            pbar.set_description(f"Testando remoção da feature: {feat_to_remove}")
            
            assembler_temp = VectorAssembler(inputCols=temp_features, outputCol="features")
            df_train_temp = assembler_temp.transform(df_train_sample)
            df_val_temp = assembler_temp.transform(df_val)
            
            gbt_temp = GBTRegressor(
                featuresCol="features",
                labelCol=label_col,
                maxIter=gbt_params.get('maxIter', 20),
                maxDepth=gbt_params.get('maxDepth', 5),
                stepSize=gbt_params.get('stepSize', 0.1),
                seed=gbt_params.get('seed', 42)
            ).fit(df_train_temp)
            
            predictions_temp = gbt_temp.transform(df_val_temp)
            wmape_train_temp = calculate_wmape(gbt_temp.transform(df_train_temp), 'prediction', label_col)
            wmape_val_temp = calculate_wmape(predictions_temp, 'prediction', label_col)
            
            current_overfitting_score = abs(wmape_val_temp / wmape_train_temp - 1)
            
            pbar.write(f"   WMAPE (Val) com remoção: {wmape_val_temp:.4f}, Overfitting Score: {current_overfitting_score:.4f}")
            
            if current_overfitting_score < best_overfitting_score:
                pbar.write(f"   Melhora na generalização! Mantendo {feat_to_remove} fora.")
                current_features = temp_features
                best_overfitting_score = current_overfitting_score
                removed_features.append(feat_to_remove)
            else:
                pbar.write("   Nenhuma melhora. Retornando feature.")
            
    final_selected_features = [f for f in feature_cols if f not in removed_features]
    
    print("\n--- Processo de Seleção Concluído ---")
    print(f"Features selecionadas: {len(final_selected_features)} de {len(feature_cols)}")
    print(f"Lista de features finais: {final_selected_features}")
    
    df_train.unpersist()
    df_val.unpersist()
    
    return final_selected_features