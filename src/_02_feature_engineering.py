import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, weekofyear, dense_rank, lag, concat, lit, countDistinct, sum, avg, min, max, stddev, when, size, collect_set, lpad, cast
from pyspark.sql.window import Window 
from datetime import datetime

# --- Funções de Engenharia de Features ---

def create_week_and_month_features(df):
    """
    Cria features de tempo como 'Semana' e 'week_of_month' usando funções nativas do Spark.
    
    Args:
        df (pyspark.sql.DataFrame): DataFrame de entrada.
        
    Returns:
        pyspark.sql.DataFrame: DataFrame com as novas features de tempo.
    """
    print("Criando features de tempo (semana do ano e semana do mês)...")
    df = df.withColumn('Semana', weekofyear(col('transaction_date')).cast('integer'))
    
    # Calculando a semana do mês com funções nativas do Spark (sem UDF)
    df = df.withColumn(
        'week_of_month',
        (dayofmonth(col('transaction_date')) - lit(1)).cast('integer') / lit(7) + lit(1)
    )
    return df

def create_global_week_id_and_rank(df):
    """
    Cria um identificador de semana global sequencial para ser usado em janelas.

    Args:
        df (pyspark.sql.DataFrame): O DataFrame contendo a coluna 'Semana'.

    Returns:
        pyspark.sql.DataFrame: O DataFrame com a nova coluna 'week_rank'.
    """
    print("Criando rank sequencial baseado na ordem das semanas...")
    window_rank_spec = Window.orderBy("Semana")
    df_with_week_rank = df.withColumn(
        "week_rank",
        dense_rank().over(window_rank_spec)
    )
    return df_with_week_rank

def calculate_lagged_features(df_with_week_rank, week_windows, value_columns, aggregation_functions):
    """
    Calcula features defasadas para diferentes janelas de semanas.

    Args:
        df_with_week_rank (pyspark.sql.DataFrame): O DataFrame com a coluna 'week_rank'.
        week_windows (list): Lista de números inteiros representando as janelas de semanas.
        value_columns (list): Lista de strings com os nomes das colunas numéricas para aplicar as agregações.
        aggregation_functions (list): Lista de strings representando as funções de agregação.

    Returns:
        pyspark.sql.DataFrame: O DataFrame original com as novas colunas de features defasadas.
    """
    print("Calculando features defasadas...")
    df_result = df_with_week_rank

    # Janela para lag-N
    lag_window_spec = Window.partitionBy("internal_store_id", "internal_product_id").orderBy("Semana")
    
    # Janela para agregados móveis (rolling aggregates)
    base_lag_window_spec = lambda window: Window.partitionBy("internal_store_id", "internal_product_id").orderBy("Semana").rowsBetween(-window, -1)
    
    for window in week_windows:
        # Lag de vendas
        df_result = df_result.withColumn(
            f'Vendas_Anteriores_lag_{window}', 
            lag(col('Vendas_Semanais'), offset=window).over(lag_window_spec)
        )
        
        for col_name in value_columns:
            for func_name in aggregation_functions:
                if func_name == 'sum':
                    df_result = df_result.withColumn(f'sum_{col_name}_last_{window}_weeks', sum(col(col_name)).over(base_lag_window_spec(window)))
                elif func_name == 'avg' or func_name == 'mean':
                    df_result = df_result.withColumn(f'avg_{col_name}_last_{window}_weeks', avg(col(col_name)).over(base_lag_window_spec(window)))
                elif func_name == 'min':
                    df_result = df_result.withColumn(f'min_{col_name}_last_{window}_weeks', min(col(col_name)).over(base_lag_window_spec(window)))
                elif func_name == 'max':
                    df_result = df_result.withColumn(f'max_{col_name}_last_{window}_weeks', max(col(col_name)).over(base_lag_window_spec(window)))
                elif func_name == 'std' or func_name == 'stddev':
                    df_result = df_result.withColumn(f'stddev_{col_name}_last_{window}_weeks', stddev(col(col_name)).over(base_lag_window_spec(window)))

    # A contagem distinta é tratada de forma mais complexa para janelas, por isso a contagem
    # de produtos distintos por semana/loja será feita separadamente
    df_result = df_result.fillna(-1)
    return df_result

def run_feature_engineering_pipeline(df, spark):
    """
    Função principal que orquestra a criação de features em uma sequência lógica.

    Args:
        df (pyspark.sql.DataFrame): DataFrame de entrada com os dados brutos.
        spark (SparkSession): Sessão Spark ativa.

    Returns:
        pyspark.sql.DataFrame: DataFrame final com todas as features.
    """
    if df is None:
        print("DataFrame de entrada para a função principal é None.")
        return None

    # Agregação para o nível de análise
    df_features = df.groupby(['internal_store_id', 'internal_product_id', 'transaction_date', 'distributor_id', 
                             'categoria', 'subcategoria', 'marca', 'fabricante', 
                             'tipos', 'label', 'descricao']).agg(
        sum('quantity').alias('Vendas_Semanais'),
        sum('gross_value').alias('gross_value'),
        sum('net_value').alias('net_value'),
        sum('gross_profit').alias('gross_profit'),
        sum('discount').alias('discount'),
        sum('taxes').alias('taxes')
    )

    # Passo 1: Criar features de tempo
    df_with_time_features = create_week_and_month_features(df_features)

    # Passo 2: Criar rank sequencial
    df_with_rank = create_global_week_id_and_rank(df_with_time_features)

    # Passo 3: Calcular features defasadas e de janela móvel
    week_windows_to_consider = [1, 3, 6, 9, 12, 15, 18]
    value_cols_to_aggregate = ['Vendas_Semanais', 'gross_value', 'net_value', 'gross_profit', 'discount', 'taxes']
    aggregation_functions_to_apply = ['sum', 'avg', 'min', 'max', 'stddev'] 

    final_df = calculate_lagged_features(
        df_with_rank,
        week_windows_to_consider,
        value_cols_to_aggregate,
        aggregation_functions_to_apply
    )
    
    return final_df

if __name__ == "__main__":
    # Inicia a sessão Spark
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    input_path = "data/processed/base_consolidada.parquet"
    
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado. Execute 01_load_data.py primeiro.")
    else:
        df_consolidado = spark.read.parquet(input_path)
        df_final = run_feature_engineering_pipeline(df_consolidado, spark)
        if df_final:
            output_path = r"data/processed/base_com_features.parquet"
            print(f"\nSalvando o DataFrame final em {output_path}...")
            df_final.write.mode("overwrite").parquet(output_path)
            print("Arquivo salvo com sucesso.")
    spark.stop()