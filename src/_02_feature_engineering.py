from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, weekofyear, dense_rank, lag, concat, lit, countDistinct, count, mean, sum, avg, min, max, stddev, when, size, collect_set, lpad, cast
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
    # 1. Adiciona a coluna 'Ano' para usar na partição
    df_with_year = df.withColumn("Ano", year(col("reference_date")))

    # 2. Cria o identificador de semana global para ordenação
    df_with_week_id = df_with_year.withColumn(
        "global_week_id",
        concat(
            col("Ano"),
            lit("-"),
            lpad(month(col("reference_date")).cast("string"), 2, "0"),
            lit("-"),
            col("week_of_month")
        )
    )

    # 3. Cria o rank sequencial, particionando por Ano para reiniciar a contagem
    window_rank_spec = Window.partitionBy("Ano").orderBy("global_week_id")
    df_with_week_rank = df_with_week_id.withColumn(
        "week_rank",
        dense_rank().over(window_rank_spec)
    )

    # Cria um rank sequencial baseado na ordem dos global_week_id distintos
    window_rank_spec = Window.orderBy("global_week_id")
    df_with_week_rank = df_with_week_rank.withColumn(
      "rank",
      dense_rank().over(window_rank_spec)
      
    )

    return df_with_week_rank

def calculate_time_since_last_order(df):
    """
    Calcula o tempo_ultimo_pedido (diferença entre a semana atual e a última venda).
    
    Args:
        df (Spark DataFrame): DataFrame com a coluna 'rank'.
        
    Returns:
        Spark DataFrame: DataFrame com a nova coluna 'tempo_ultimo_pedido'.
    """
    print("Calculando tempo_ultimo_pedido...")
    
    window_spec = Window.partitionBy("internal_store_id", "internal_product_id").orderBy("rank")
    
    # Encontra a semana do último pedido para cada linha
    df_with_last_order = df.withColumn(
        "last_order_week",
        lag(col("rank"), offset=1).over(window_spec)
    )
    
    # Calcula a diferença entre a semana atual e a última semana com pedido
    df_with_time_since_last_order = df_with_last_order.withColumn(
        "tempo_ultimo_pedido",
        when(
            col("last_order_week").isNull(),
            -1
        ).otherwise(
            col("rank") - col("last_order_week")
        )
    )
    
    return df_with_time_since_last_order

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
    
    df_result = df.filter((col('quantity') > 0) | isnan(col('quantity')))
    
    # Adicionando a contagem distinta de produtos por loja por semana
    if 'count' in aggregation_functions:
        # A sua lógica: criar um dataframe auxiliar com a contagem distinta por semana e loja
        print("Calculando contagem distinta de produtos por semana...")
        df_distinct_products_weekly = df_result.groupby('internal_store_id', 'rank').agg(
            count('internal_product_id').alias('distinct_products_count_weekly')
        ).dropDuplicates(['internal_store_id', 'rank'])
        
        # A correção principal: calculamos a janela móvel neste dataframe auxiliar
        # antes de fazer o join, garantindo que o calculo seja preciso
        distinct_products_window_spec = lambda window: Window.partitionBy('internal_store_id').orderBy(col('rank')).rangeBetween(-window, -1)
        
        for window in week_windows:
            final_distinct_count = when(
                col('rank') <= window,
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
            on=['internal_store_id', 'rank'], 
            how='left'
        )

    # Janela para lag-N e agregados móveis
    base_window_spec = lambda window: Window.partitionBy("internal_store_id", "internal_product_id").orderBy(col('rank')).rangeBetween(-window, -1)
    lag_window_spec = Window.partitionBy("internal_store_id", "internal_product_id").orderBy(col('rank'))
    
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
                        col('rank') <= window,
                        lit(None)
                    ).otherwise(
                        agg_expr.over(base_window_spec(window))
                    )
                    expressions.append(final_agg_expr.alias(f'{func_name}_{col_name}_last_{window}_weeks'))
    
    df_with_features = df_result.select(expressions)
    df_with_features = df_with_features.fillna(-1) 
    
    return df_with_features

def aggregate_dataframe(df, ignore_cols: list, agg_ops):
    """
    Agrega um DataFrame Spark por todas as colunas, exceto as da lista de ignorar.
    As operações de agregação são definidas por uma lista ou um dicionário.

    Args:
        df (DataFrame): O DataFrame Spark de entrada.
        ignore_cols (list): Uma lista de nomes de colunas para excluir das chaves de agrupamento.
        agg_ops (list or dict): Uma lista de colunas para aplicar uma agregação padrão,
                                ou um dicionário de colunas com uma lista de funções.

    Returns:
        DataFrame: O DataFrame Spark agregado.
    """
    all_cols = df.columns
    # As chaves de agrupamento são todas as colunas que não estão em ignore_cols e não são as colunas a serem agregadas
    if isinstance(agg_ops, list):
        agg_cols = agg_ops
    elif isinstance(agg_ops, dict):
        agg_cols = list(agg_ops.keys())
    else:
        raise ValueError("agg_ops deve ser uma lista ou um dicionário.")
    
    group_cols = [c for c in all_cols if c not in ignore_cols and c not in agg_cols]

    agg_exprs = []
    
    if isinstance(agg_ops, list):
        # Lógica para o formato de lista (mantém a estrutura original)
        agg_func_type = 'sum' # Exemplo de função padrão
        for col_name in agg_ops:
            if col_name in all_cols:
                agg_exprs.append(sum(col(col_name)).alias(f'{col_name}'))
    else:
        # Lógica para o formato de dicionário (nova funcionalidade)
        for col_name, func_list in agg_ops.items():
            if col_name in all_cols:
                for func_name in func_list:
                    if func_name.lower() == 'sum':
                        agg_exprs.append(sum(col(col_name)).alias(f'{col_name}'))
                    elif func_name.lower() == 'mean':
                        agg_exprs.append(mean(col(col_name)).alias(f'mean_{col_name}'))
                    elif func_name.lower() == 'min':
                        agg_exprs.append(min(col(col_name)).alias(f'min_{col_name}'))
                    elif func_name.lower() == 'count':
                        agg_exprs.append(count(col(col_name)).alias(f'quantidade_pedidos'))
                    else:
                        print(f"Aviso: Função de agregação '{func_name}' não suportada para coluna '{col_name}'.")
            else:
                print(f"Aviso: Coluna '{col_name}' não encontrada no DataFrame.")

    if not group_cols:
        if agg_exprs:
            return df.agg(*agg_exprs)
        else:
            return df
    else:
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

    df_with_week_ids_and_rank = create_global_week_id_and_rank(df)

    # Passo 2: Agrupar informações a nivel semanal

    ignore_list = [
        'descricao', 
        'distributor_id', 
        'pdv', 
        'produto',
        'transaction_date',
        'Vendas_Semanais'
    ]
    aggregate_list = [
        'quantity'
    ]
    aggregation_type = 'sum'

    agg_dict = {"quantity": ['sum', 'count']}

    aggregated_df = aggregate_dataframe(
        df=df_with_week_ids_and_rank,
        ignore_cols=ignore_list,
        agg_ops=agg_dict
    )

    df_with_time_since_order = calculate_time_since_last_order(aggregated_df)

    # Passo 3: Calcular features defasadas usando o rank da semana
    final_df_with_lagged_features = calculate_lagged_features_optimized(
        df_with_time_since_order,
        week_windows,
        value_columns,
        aggregation_functions
    )

    return final_df_with_lagged_features