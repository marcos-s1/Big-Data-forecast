from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, weekofyear, dense_rank, lag, concat, lit, countDistinct, count, sum, avg, min, max, stddev, when, size, collect_set, lpad, cast
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
import os
from pyspark.sql.window import Window 
from datetime import datetime

def calculate_week_of_month(transaction_date, reference_date):
    """
    Calculates the week number within the month for a given transaction date.

    Args:
        transaction_date (date): The date of the transaction (YYYY-MM-DD).
        reference_date (date): A reference date within the month (YYYY-MM-DD).

    Returns:
        int: The week number within the month (starting from 1).
    """
    # Ensure dates are datetime objects
    if isinstance(transaction_date, str):
        transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d').date()
    if isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, '%Y-%m-%d').date()

    # Convert reference_date to the first day of the month
    first_day_of_month = reference_date.replace(day=1)

    # Calculate the difference in days
    day_difference = (transaction_date - first_day_of_month).days

    # Calculate the week number (starting from 1)
    week_number = (day_difference // 7) + 1

    return week_number

def create_global_week_id_and_rank(df):
    """
    Cria um identificador de semana global sequencial (ano, mês, semana do mês)
    e um rank sequencial baseado na ordem das semanas.

    Args:
        df (Spark DataFrame): O DataFrame contendo as colunas 'reference_date' e 'week_of_month'.

    Returns:
        Spark DataFrame: O DataFrame com as novas colunas 'global_week_id' e 'week_rank'.
    """
    print("Criando identificador de semana global e rank...")
    # Cria um identificador baseado no ano, mês e semana do mês para ordenação
    # Formata o mês com zero à esquerda (MM) e concatena as partes
    df_with_week_id = df.withColumn(
        "global_week_id",
        concat(
            year(col("reference_date")),
            lit("-"),
            lpad(month(col("reference_date")).cast("string"), 2, "0"), # Formata o mês com 2 dígitos
            lit("-"),
            col("week_of_month")
        )
    )

    # Cria um rank sequencial baseado na ordem dos global_week_id distintos
    window_rank_spec = Window.orderBy("global_week_id")
    df_with_week_rank = df_with_week_id.withColumn(
        "week_rank",
        dense_rank().over(window_rank_spec)
    )
    return df_with_week_rank

def calculate_lagged_features_optimized(df, week_windows, value_columns, aggregation_functions):
    """
    Calcula features defasadas e de janela móvel de forma otimizada.

    Args:
        df (Spark DataFrame): O DataFrame de entrada.
        week_windows (list): Lista de números inteiros representando as janelas de semanas.
        value_columns (list): Lista de strings com os nomes das colunas numéricas para aplicar as agregações.
        aggregation_functions (list): Lista de strings representando as funções de agregação.

    Returns:
        Spark DataFrame: O DataFrame original com as novas colunas de features defasadas.
    """
    print("Calculando features defasadas e de janela móvel de forma otimizada...")
    
    df_result = df
    
    # Adicionando a contagem distinta de produtos por loja por semana
    if 'count' in aggregation_functions:
        # A sua lógica: criar um dataframe auxiliar com a contagem distinta por semana e loja
        print("Calculando contagem distinta de produtos por semana...")
        df_distinct_products_weekly = df_result.groupby('pdv', 'week_rank').agg(
            count('produto').alias('distinct_products_count_weekly')
        ).dropDuplicates(['pdv', 'week_rank'])
        
        # A correção principal: calculamos a janela móvel neste dataframe auxiliar
        # antes de fazer o join, garantindo que o calculo seja preciso
        distinct_products_window_spec = lambda window: Window.partitionBy('pdv').orderBy(col('week_rank')).rangeBetween(-window, -1)
        
        for window in week_windows:
            final_distinct_count = when(
                col('week_rank') <= window,
                lit(None)
            ).otherwise(
                sum(col('distinct_products_count_weekly')).over(distinct_products_window_spec(window))
            )
            df_distinct_products_weekly = df_distinct_products_weekly.withColumn(
                f'distinct_products_last_{window}_weeks', final_distinct_count
            )
        
        # Agora, fazemos o join deste dataframe auxiliar com o principal
        df_result = df_result.join(
            df_distinct_products_weekly.drop('distinct_products_count_weekly'), 
            on=['pdv', 'week_rank'], 
            how='left'
        )

    # Janela para lag-N e agregados móveis
    base_window_spec = lambda window: Window.partitionBy("pdv", "produto").orderBy(col('week_rank')).rangeBetween(-window, -1)
    lag_window_spec = Window.partitionBy("pdv", "produto").orderBy(col('week_rank'))
    
    expressions = []
    
    original_cols = [c for c in df_result.columns if 'distinct_products' not in c]
    expressions.extend([col(c) for c in original_cols])
    
    # Adicionamos as colunas de contagem distintas que acabamos de criar
    distinct_cols = [c for c in df_result.columns if 'distinct_products' in c]
    expressions.extend([col(c) for c in distinct_cols])

    for window in week_windows:
        expressions.append(
            lag(col('quantity'), offset=window).over(lag_window_spec).alias(f'Vendas_Anteriores_lag_{window}')
        )

    for window in week_windows:
        for col_name in value_columns:
            for func_name in aggregation_functions:
                agg_expr = None
                if func_name == 'sum':
                    agg_expr = sum(col(col_name))
                elif func_name == 'avg' or func_name == 'mean':
                    agg_expr = avg(col(col_name))
                elif func_name == 'min':
                    agg_expr = min(col(col_name))
                elif func_name == 'max':
                    agg_expr = max(col(col_name))
                elif func_name == 'std' or func_name == 'stddev':
                    agg_expr = stddev(col(col_name))

                if agg_expr is not None:
                    final_agg_expr = when(
                        col('week_rank') <= window,
                        lit(None)
                    ).otherwise(
                        agg_expr.over(base_window_spec(window))
                    )
                    expressions.append(final_agg_expr.alias(f'{func_name}_{col_name}_last_{window}_weeks'))
    
    df_with_features = df_result.select(expressions)
    df_with_features = df_with_features.fillna(-1) 
    
    return df_with_features

def aggregate_dataframe(df, ignore_cols: list, agg_cols: list, agg_func_type: str):
    """
    Aggregates a Spark DataFrame by all columns except those in ignore_cols,
    applying a specified aggregation function to columns in agg_cols.

    Args:
        df (DataFrame): The input Spark DataFrame.
        ignore_cols (list): A list of column names to exclude from the grouping keys.
        agg_cols (list): A list of column names to apply the aggregation function to.
        agg_func_type (str): The type of aggregation function ('mean', 'sum', 'min').

    Returns:
        DataFrame: The aggregated Spark DataFrame.
    """
    # Get all column names from the DataFrame
    all_cols = df.columns

    # Determine the grouping columns (all columns not in ignore_cols)
    group_cols = [c for c in all_cols if c not in ignore_cols and c not in agg_cols]

    # Prepare the aggregation expressions
    agg_exprs = []
    for col_name in agg_cols:
        if col_name in all_cols:
            if agg_func_type.lower() == 'mean':
                agg_exprs.append(mean(col(col_name)).alias(f'{col_name}'))
            elif agg_func_type.lower() == 'sum':
                agg_exprs.append(sum(col(col_name)).alias(f'{col_name}'))
            elif agg_func_type.lower() == 'min':
                agg_exprs.append(min(col(col_name)).alias(f'{col_name}'))
            else:
                print(f"Warning: Unsupported aggregation function type '{agg_func_type}'. Skipping column '{col_name}'.")
        else:
             print(f"Warning: Aggregation column '{col_name}' not found in DataFrame. Skipping.")

    if not group_cols:
        print("Warning: No grouping columns determined. Performing a global aggregation.")
        if agg_exprs:
            return df.agg(*agg_exprs)
        else:
            print("No aggregation expressions defined.")
            return df
    elif not agg_exprs:
        print("Warning: No valid aggregation expressions determined. Returning DataFrame grouped by group_cols without aggregations.")
        return df.select(group_cols).distinct()
    else:
        print(f"Grouping by: {group_cols}")
        print(f"Applying '{agg_func_type}' aggregation to: {agg_cols}")
        return df.groupBy(group_cols).agg(*agg_exprs)

def feature_engineering_pipeline(df, week_windows, value_columns, aggregation_functions, spark_session):
    """
    Função principal para calcular features defasadas para o modelo.

    Args:
        df (Spark DataFrame): O DataFrame de entrada contendo dados de transação e PDV.
        week_windows (list): Lista de números inteiros representando as janelas de semanas (ex: [1, 3, 6, 9, 12, 15, 18]).
        value_columns (list): Lista de strings com os nomes das colunas numéricas para aplicar as agregações
                              (ex: ['quantity', 'gross_value']).
        aggregation_functions (list): Lista de strings representando as funções de agregação a serem aplicadas
                                     (ex: ['min', 'max', 'sum', 'mean', 'std', 'count']). 'count' é aplicado a distinct_products.

    Returns:
        Spark DataFrame: O DataFrame com as features defasadas calculadas.
    """
    if df is None:
        print("DataFrame de entrada para a função main é None.")
        return None

    # Passo 1: Criar semana do mes em questão
    # Define the Spark UDF
    calculate_week_of_month_udf = udf(calculate_week_of_month, IntegerType())

    # Apply the UDF to the DataFrame to create the new column
    df = df.withColumn(
        'week_of_month',
        calculate_week_of_month_udf(col('transaction_date'), col('reference_date'))
    )

    # Passo 1: Criar identificador de semana global e rank
    df_with_week_ids_and_rank = create_global_week_id_and_rank(df)

    # Passo 2: Agrupar informações a nivel semanal

    ignore_list = [
        'descricao', 
        'distributor_id', 
        'internal_store_id', 
        'internal_product_id',
        'transaction_date',
        'Vendas_Semanais'
    ]
    aggregate_list = [
        'quantity', 
        'gross_value',
        'net_value', 
        'gross_profit', 
        'discount', 
        'taxes',
    ]
    aggregation_type = 'sum'

    aggregated_df = aggregate_dataframe(
    df=df_with_week_ids_and_rank,
    ignore_cols=ignore_list,
    agg_cols=aggregate_list,
    agg_func_type=aggregation_type
    )

    # Passo 3: Calcular features defasadas usando o rank da semana
    final_df_with_lagged_features = calculate_lagged_features_optimized(
        df_with_week_ids_and_rank,
        week_windows,
        value_columns,
        aggregation_functions
    )

    return final_df_with_lagged_features