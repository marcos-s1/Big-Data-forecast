import os
import pickle
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, lit, when

def group_low_frequency_categories(df: DataFrame, categorical_cols: list, threshold: float = 0.01) -> tuple[DataFrame, dict]:
    """
    Agrupa categorias de baixa frequência em colunas de um DataFrame Spark em uma categoria 'Outros'.

    Args:
        df (DataFrame): O DataFrame Spark de entrada.
        categorical_cols (list): Uma lista de nomes de colunas para processar.
        threshold (float): O limite de frequência abaixo do qual as categorias serão agrupadas.

    Returns:
        tuple[DataFrame, dict]: Um DataFrame Spark processado e um dicionário
                                com as categorias agrupadas para cada coluna.
    """
    df_processed = df
    grouped_categories_map = {}

    total_rows = df.count()
    if total_rows == 0:
        print("DataFrame vazio. Nenhuma ação a ser tomada.")
        return df, {}
    
    for c in categorical_cols:
        if c in df_processed.columns:
            print(f"Processando a coluna: {c}")

            category_counts = df_processed.groupBy(c).agg(count(c).alias("count"))
            
            low_freq_categories_df = category_counts.withColumn(
                "frequency",
                col("count") / lit(total_rows)
            ).filter(col("frequency") < threshold)
            
            low_freq_list = [row[c] for row in low_freq_categories_df.collect()]

            if low_freq_list:
                print(f"  Agrupando {len(low_freq_list)} categorias com frequência < {threshold:.2%} em 'Outros' para a coluna '{c}'")
                
                grouped_categories_map[c] = low_freq_list
                
                df_processed = df_processed.withColumn(
                    c,
                    when(col(c).isin(low_freq_list), lit("Outros")).otherwise(col(c))
                )
            else:
                print(f"  Nenhuma categoria com frequência < {threshold:.2%} para agrupar na coluna '{c}'")
        else:
            print(f"Aviso: Coluna '{c}' não encontrada no DataFrame.")

    return df_processed, grouped_categories_map