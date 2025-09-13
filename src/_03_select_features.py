import os
import shutil
import tempfile
import pandas as pd # Necessário para o DataFrame temporário de resultados
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, abs, sum as spark_sum, lit
from pyspark.ml.feature import VectorAssembler
from catboost.spark import CatBoostRegressor
from tqdm.auto import tqdm

# --- CONFIGURAÇÃO DO MODELO ---
TARGET = 'quantity'
IDENTIFIERS = ['pdv', 'produto']
IGNORE_FEATURES = [
      'internal_store_id',
      'internal_product_id',
      'transaction_date',
      'reference_date',
      'week_of_month',
      'Vendas_Semanais'
      ]
CATEGORICAL_FEATURES = [
    'premise',
    'categoria_pdv',
    'zipcode',
    'categoria',
    'tipos',
    'label',
    'subcategoria',
    'marca',
    'fabricante',
    'week_of_month',
    'global_week_id',
]
WEEKS_FOR_VALIDATION = 4

# --- Funções Auxiliares ---
def wmape_pyspark(df, y_true_col, y_pred_col):
    """Calcula o WMAPE para um DataFrame Spark."""
    total_abs_error = df.withColumn("abs_error", abs(col(y_true_col) - col(y_pred_col))).agg(spark_sum("abs_error")).collect()[0][0]
    total_actual = df.agg(spark_sum(y_true_col)).collect()[0][0]
    
    if total_actual == 0:
        return 0.0
    return total_abs_error / total_actual if total_actual else 0.0

def clean_and_prepare_spark_df(df, categorical_features):
    """
    Limpa e prepara o DataFrame Spark para modelagem, lidando com NaNs.
    """
    # Lida com NaNs em features numéricas
    numerical_cols = [c for c in df.columns if c not in categorical_features + [TARGET] + IDENTIFIERS]
    for c in numerical_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
        
    # Lida com NaNs em features categóricas
    for c in categorical_features:
        df = df.withColumn(c, when(col(c).isNull(), lit("Missing_Category")).otherwise(col(c)))
        
    return df

def run_feature_selection_pyspark(spark, df, target, identifiers, ignore_features, categorical_features, weeks_for_validation):
    """
    Executa a seleção de features usando backward elimination e WMAPE em Spark.
    """
    print("--- 1. Preparando o DataFrame Spark para a seleção de features ---")
    
    # Define o conjunto de features a serem avaliadas
    all_features = [c for c in df.columns if c not in [target] + identifiers + ignore_features]
    
    # Cria o conjunto de treino e validação (out-of-time)
    max_week_rank = df.agg({"week_rank": "max"}).collect()[0][0]
    cutoff_week = max_week_rank - weeks_for_validation
    df_train = df.filter(col('week_rank') <= cutoff_week)
    df_val = df.filter(col('week_rank') > cutoff_week)

    print(f"Conjunto de treino: {df_train.count()} linhas")
    print(f"Conjunto de validação: {df_val.count()} linhas")
    
    # --- 2. Backward Selection (Eliminação Reversa) ---
    print("\n--- 2. Iniciando Backward Selection (Pode levar tempo) ---")

    current_features = list(all_features)
    removed_features = []
    best_wmape_val = 1e9

    catboost_params = {
        'iterations': 200, 
        'learningRate': 0.05, 
        'depth': 6, 
        'randomSeed': 42,
        'earlyStoppingRounds': 25,
        'l2LeafRegularizer': 3,
        'lossFunction': 'RMSE'
    }

    print('Total de Features Inicialmente:', len(current_features))
    
    with tqdm(total=len(current_features) - 1, desc="Backward Selection") as pbar_outer:
        while len(current_features) > 1:
            wmape_results = []
            
            pbar_inner = tqdm(current_features, desc=f"Testando {len(current_features)} features")
            for feature_to_remove in pbar_inner:
                test_features = [f for f in current_features if f != feature_to_remove]
                
                # Vetoriza as features para o modelo de ML do Spark
                assembler = VectorAssembler(inputCols=test_features, outputCol="features_vector")
                df_train_vector = assembler.transform(df_train)
                df_val_vector = assembler.transform(df_val)

                model = CatBoostRegressor(**catboost_params)
                
                # Para usar o early stopping no CatBoost Spark, você deve passar o eval_set no fit
                model.fit(df_train_vector, eval_set=df_val_vector)
                
                predictions_train = model.transform(df_train_vector)
                predictions_val = model.transform(df_val_vector)
                
                wmape_train = wmape_pyspark(predictions_train, target, 'prediction')
                wmape_val = wmape_pyspark(predictions_val, target, 'prediction')
                
                # Validação de Overfitting
                if abs(wmape_val / wmape_train - 1) > 0.10:
                    pbar_outer.write(f"Aviso: Overfitting detectado ao remover '{feature_to_remove}'. WMAPE de treino vs. validação: {wmape_train:.4f} vs {wmape_val:.4f}.")
                    continue 

                wmape_results.append({'feature_to_remove': feature_to_remove, 'wmape_val': wmape_val, 'wmape_train': wmape_train})
            
            pbar_inner.close()
            
            if not wmape_results:
                pbar_outer.write("\nNenhuma feature atendeu aos critérios de WMAPE. Parando a eliminação.")
                break

            # Otimização Multi-Critério
            wmape_results_df = pd.DataFrame(wmape_results)
            wmape_results_df['diff_abs'] = abs(wmape_results_df['wmape_val'] - wmape_results_df['wmape_train'])
            wmape_results_df = wmape_results_df.sort_values(by=['wmape_val', 'diff_abs'])
            
            best_removal_row = wmape_results_df.iloc[0]
            feature_to_remove = best_removal_row['feature_to_remove']
            wmape_after_removal = best_removal_row['wmape_val']
            
            if wmape_after_removal < best_wmape_val:
                best_wmape_val = wmape_after_removal
                removed_features.append(feature_to_remove)
                current_features.remove(feature_to_remove)
                pbar_outer.write(f"-> Removida: '{feature_to_remove}'. Novo melhor WMAPE: {best_wmape_val:.4f} (diff: {best_removal_row['diff_abs']:.4f})")
            else:
                pbar_outer.write("\nO WMAPE não melhorou mais. Parando a eliminação.")
                break
            
            pbar_outer.update(1)
            
    final_features = current_features
    print("\n--- Backward Selection Finalizada ---")
    print(f"Conjunto de features selecionado: {final_features}")
    print(f"WMAPE alcançado: {best_wmape_val:.4f}")
    
    return final_features

if __name__ == "__main__":
    spark = SparkSession.builder.appName("FeatureSelection").getOrCreate()
    input_path = "data/processed/base_com_features.parquet"
    
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado. Execute 02_feature_engineering.py primeiro.")
    else:
        df_com_features_spark = spark.read.parquet(input_path)
        
        df_cleaned = clean_and_prepare_spark_df(df_com_features_spark, CATEGORICAL_FEATURES)
        
        features_selecionadas = run_feature_selection_pyspark(
            spark, df_cleaned, TARGET, IDENTIFIERS, IGNORE_FEATURES, CATEGORICAL_FEATURES, WEEKS_FOR_VALIDATION
        )
        
        output_path = "models/features_selecionadas.txt"
        with open(output_path, "w") as f:
            for item in features_selecionadas:
                f.write(f"{item}\n")
        print(f"\nLista de features selecionadas salva em '{output_path}'.")
        
    spark.stop()