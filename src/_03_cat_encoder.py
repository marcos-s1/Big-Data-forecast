import os
import shutil
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

def train_and_save_encoder(spark: SparkSession, df, categorical_cols: list, output_model_path: str):
    """
    Treina e salva um pipeline de codificação para colunas categóricas.
    """
    print("--- Treinando e salvando o modelo de codificação ---")
    
    indexed_features = [c + "_indexed" for c in categorical_cols]
    encoded_features = [c + "_encoded" for c in categorical_cols]

    indexers = [
        StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep")
        for c, idx in zip(categorical_cols, indexed_features)
    ]

    encoders = [
        OneHotEncoder(inputCol=idx, outputCol=enc)
        for idx, enc in zip(indexed_features, encoded_features)
    ]
    
    pipeline = Pipeline(stages=indexers + encoders)
    
    print("Treinando o pipeline...")
    pipeline_model = pipeline.fit(df)
    
    try:
        if os.path.exists(output_model_path):
            shutil.rmtree(output_model_path)
            print(f"Diretório de modelo existente removido: {output_model_path}")

        pipeline_model.save(output_model_path)
        print(f"Modelo de codificação salvo com sucesso em: {output_model_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")
        return False

def load_and_transform_data(spark: SparkSession, df, model_path: str, categorical_features: list):
    """
    Carrega um modelo de codificação salvo e o aplica a um novo DataFrame.
    
    Args:
        spark (SparkSession): Sessão Spark ativa.
        df (pyspark.sql.DataFrame): DataFrame de entrada.
        model_path (str): Caminho para o modelo de codificação salvo.
        categorical_features (list): Lista de colunas categóricas que foram codificadas.
        
    Returns:
        pyspark.sql.DataFrame: DataFrame com as colunas one-hot encoded e sem as originais.
    """
    print("--- Carregando o modelo e aplicando a transformação ---")
    
    try:
        # Carrega o modelo de codificação treinado
        loaded_model = PipelineModel.load(model_path)
        print(f"Modelo carregado com sucesso de: {model_path}")
        
        # Aplica o modelo para transformar o novo DataFrame
        df_transformed = loaded_model.transform(df)
        
        # Remove as colunas categóricas originais e as colunas indexadas temporárias
        columns_to_drop = categorical_features + [c + "_indexed" for c in categorical_features]
        df_transformed = df_transformed.drop(*columns_to_drop)
        
        print("\nDataFrame transformado com sucesso. Schema:")
        df_transformed.printSchema()
        df_transformed.show(5, truncate=False)
        
        return df_transformed
    except Exception as e:
        print(f"Erro ao carregar ou aplicar o modelo: {e}")
        return None