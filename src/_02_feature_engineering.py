from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, weekofyear, dense_rank, lag, concat, lit, countDistinct, sum, avg, min, max, stddev, when, size, collect_set, lpad, cast
from pyspark.sql.window import Window # Importar a classe Window
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType

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

def calculate_lagged_features(df_with_week_rank, week_windows, value_columns, aggregation_functions):
    """
    Calcula features defasadas (lagged features) para diferentes janelas de semanas,
    colunas de valor e funções de agregação, usando o rank da semana para definir
    a janela e garantindo valores -1 para as primeiras semanas.

    Args:
        df_with_week_rank (Spark DataFrame): O DataFrame com as colunas 'global_week_id' e 'week_rank'.
        week_windows (list): Lista de números inteiros representando as janelas de semanas (ex: [1, 3, 6]).
        value_columns (list): Lista de strings com os nomes das colunas numéricas para aplicar as agregações
                              (ex: ['quantity', 'gross_value']).
        aggregation_functions (list): Lista de strings representando as funções de agregação a serem aplicadas
                                     (exclui 'count' que é específico para distinct_products). Suportadas:
                                     'sum', 'avg', 'min', 'max', 'stddev'.

    Returns:
        Spark DataFrame: O DataFrame original com as novas colunas de features defasadas.
    """
    print("Calculando features defasadas...")
    df_result = df_with_week_rank

    # Define a janela base para o cálculo defasado usando rangeBetween no week_rank.
    # rangeBetween(-window, -1) define uma janela que inclui todos os ranks de 'week_rank - window' até 'week_rank - 1'.
    # Esta janela opera sobre o conjunto de linhas dentro do partitionBy e orderBy.
    # A janela é definida uma vez e usada para diferentes agregações e janelas de tempo.
    base_lag_window_spec = lambda window: Window.partitionBy("internal_store_id").orderBy("week_rank").rangeBetween(-window, -1)

    # Lista para armazenar os nomes das colunas defasadas criadas
    lagged_columns_created = []

    # --- Calcular Contagem Distinta de Produtos nas Janelas Defasadas ('count') ---
    # Este é tratado separadamente pois a agregação é sempre em internal_product_id
    if 'count' in aggregation_functions:
        print("Calculando contagem distinta de produtos defasada...")
        for window in week_windows:
            distinct_col_name = f"distinct_products_last_{window}_weeks"
            collected_col_name = f"collected_products_last_{window}_weeks_temp"

            # --- Nota sobre este cálculo (Contagem Distinta em Janela com Frame): ---
            # Calcular countDistinct diretamente em window functions com frames (rangeBetween/rowsBetween)
            # é uma limitação conhecida do Spark e geralmente causa AnalysisException (DISTINCT_WINDOW_FUNCTION_UNSUPPORTED).
            # A abordagem com size(collect_set(...)) é um workaround comum, mas também pode falhar com MISSING_GROUP_BY
            # devido a problemas de planejamento interno do Spark com esta combinação em alguns cenários.
            # A lógica abaixo tenta implementar o cálculo da contagem distinta sobre a janela defasada
            # (distintos produtos em TODAS as linhas da janela defasada), mas PODE resultar em erro
            # dependendo do ambiente Spark e dados. Uma alternativa mais robusta para obter métricas
            # de janela de tempo com agregação distinta NO NÍVEL SEMANAL é agregar a contagem
            # distinta semanalmente PRIMEIRO (countDistinct por semana) e depois aplicar funções
            # de janela (como sum) nos totais semanais agregados (abordagem em 2 etapas).
            # A lógica abaixo tenta calcular a contagem distinta sobre a janela defasada no nível da transação.

            # Use collect_set in the window function and then size in a chained operation
            # Esta é a tentativa de calcular o distinct count sobre a janela.
            df_result = df_result.withColumn(
                collected_col_name,
                collect_set("internal_product_id").over(base_lag_window_spec(window))
            ).withColumn(
                distinct_col_name,
                size(col(collected_col_name)) # Calculate size after the window function
            ).drop(collected_col_name) # Drop the intermediate column

            # Garantir que as primeiras 'window' semanas tenham valor -1
            df_result = df_result.withColumn(
                distinct_col_name,
                when(col("week_rank") <= window, -1).otherwise(col(distinct_col_name))
            )
            lagged_columns_created.append(distinct_col_name)

        # Remove 'count' da lista de aggregation_functions para o próximo loop
        aggregation_functions_for_values = [f for f in aggregation_functions if f != 'count']
    else:
        aggregation_functions_for_values = aggregation_functions


    # --- Calcular Outras Agregações para as Colunas de Valor Especificadas ---
    if aggregation_functions_for_values and value_columns:
        print("Calculando agregações defasadas para colunas de valor...")
        for window in week_windows:
            for col_name_to_agg in value_columns:
                for func_name in aggregation_functions_for_values:
                    output_col_name = None
                    if func_name == 'sum':
                        output_col_name = f"sum_{col_name_to_agg}_last_{window}_weeks"
                        df_result = df_result.withColumn(
                            output_col_name,
                            sum(col(col_name_to_agg)).over(base_lag_window_spec(window))
                        )
                    elif func_name == 'avg' or func_name == 'mean':
                         output_col_name = f"avg_{col_name_to_agg}_last_{window}_weeks"
                         df_result = df_result.withColumn(
                            output_col_name,
                            avg(col(col_name_to_agg)).over(base_lag_window_spec(window))
                        )
                    elif func_name == 'min':
                         output_col_name = f"min_{col_name_to_agg}_last_{window}_weeks"
                         df_result = df_result.withColumn(
                            output_col_name,
                            min(col(col_name_to_agg)).over(base_lag_window_spec(window))
                        )
                    elif func_name == 'max':
                        output_col_name = f"max_{col_name_to_agg}_last_{window}_weeks"
                        df_result = df_result.withColumn(
                            output_col_name,
                            max(col(col_name_to_agg)).over(base_lag_window_spec(window))
                        )
                    elif func_name == 'std' or func_name == 'stddev':
                         output_col_name = f"stddev_{col_name_to_agg}_last_{window}_weeks"
                         df_result = df_result.withColumn(
                            output_col_name,
                            stddev(col(col_name_to_agg)).over(base_lag_window_spec(window))
                        )
                    else:
                        print(f"Aviso: Função de agregação '{func_name}' não suportada para colunas de valor.")
                        continue # Pula para a próxima função se não for suportada

                    # Garantir que as primeiras 'window' semanas tenham valor -1
                    if output_col_name: # Verifica se a coluna foi realmente criada
                         df_result = df_result.withColumn(
                            output_col_name,
                            when(col("week_rank") <= window, -1).otherwise(col(output_col_name))
                         )
                         lagged_columns_created.append(output_col_name)
    elif aggregation_functions_for_values and not value_columns:
        print("Aviso: Funções de agregação de valor especificadas, mas nenhuma coluna de valor fornecida.")


    # Preenche quaisquer NULOS remanescentes (que não foram definidos como -1 pela condição 'when') com -1
    # Isso pode acontecer por exemplo se uma loja não tiver NENHUMA transação em uma semana que deveria estar na janela.
    # A lista de colunas a preencher deve ser as colunas defasadas que foram realmente criadas.
    df_result = df_result.fillna(-1, subset=lagged_columns_created)


    return df_result

def main(df, week_windows, value_columns, aggregation_functions, spark_session):
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

    # Passo 1: Criar identificador de semana global e rank
    df_with_week_ids_and_rank = create_global_week_id_and_rank(df)

    # Passo 2: Calcular features defasadas usando o rank da semana
    final_df_with_lagged_features = calculate_lagged_features(
        df_with_week_ids_and_rank,
        week_windows,
        value_columns,
        aggregation_functions
    )

    return final_df_with_lagged_features