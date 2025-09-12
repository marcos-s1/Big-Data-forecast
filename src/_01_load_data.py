import os
import pandas as pd
import zipfile
import tempfile
import shutil # Adicionado para limpar a pasta temporária

def read_and_join_dataframes(zip_path):
    """
    Descompacta um arquivo .zip contendo .parquet, lê os arquivos com Pandas
    e realiza o join das bases.

    Args:
        zip_path (str): Caminho para o arquivo .zip.

    Returns:
        DataFrame: O DataFrame Pandas final com as bases unidas.
    """
    print("--- Descompactando arquivos .parquet do .zip ---")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except FileNotFoundError:
        print(f"Erro: O arquivo .zip em '{zip_path}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao descompactar o arquivo: {e}")
        return None

    # O ajuste principal: agora o código procura os arquivos dentro da subpasta criada
    subfolder_path = os.path.join(temp_dir, 'hackathon_2025_templates')
    parquet_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.parquet')]
    parquet_files.sort() # Garante uma ordem consistente para o join

    if not parquet_files:
        print(f"Nenhum arquivo .parquet encontrado na subpasta. Finalizando.")
        return None

    # Dicionário para armazenar os DataFrames Pandas
    pandas_dfs = {}

    # Processa cada arquivo e armazena no dicionário
    print("--- Lendo arquivos .parquet com Pandas ---")
    for i, file in enumerate(parquet_files):
        try:
            pandas_df = pd.read_parquet(file)
            pandas_dfs[f'df_{i+1}'] = pandas_df
            print(f"Arquivo '{os.path.basename(file)}' lido com sucesso.")
        except Exception as e:
            print(f"Erro ao ler o arquivo '{os.path.basename(file)}': {e}")

    print("\n--- Realizando o join das bases ---")
    joined_df = None

    print("\nDataFrames carregados:")
    for key in pandas_dfs:
        print(f"{key}: {pandas_dfs[key].shape[0]} linhas, {pandas_dfs[key].shape[1]} colunas")
        print(pandas_dfs[key].head(2))  # Mostra as primeiras 2 linhas de cada DataFrame

    if 'df_1' in pandas_dfs and 'df_2' in pandas_dfs:
        joined_df = pd.merge(
            pandas_dfs['df_1'],
            pandas_dfs['df_2'],
            how="right",
            left_on="pdv",
            right_on="internal_store_id"
        )
        print("\nJoin de 'df_1' e 'df_2' realizado com sucesso.")

        if 'df_3' in pandas_dfs:
            joined_df = pd.merge(
                joined_df,
                pandas_dfs['df_3'],
                how="left",
                left_on="internal_product_id",
                right_on="produto"
            )
            print("Join com 'df_3' realizado com sucesso.")
        else:
            print("\nDataFrame 'df_3' não encontrado. O segundo join não foi realizado.")
    else:
        print("\nNão foi possível realizar o join. Verifique se os DataFrames foram carregados corretamente.")
        return None

    # Limpa a pasta temporária
    shutil.rmtree(temp_dir)
    print(f"\nPasta temporária '{temp_dir}' removida.")

    return joined_df

if __name__ == "__main__":
    zip_file_path = "data/raw/hackathon_2025_templates.zip"
    final_df = read_and_join_dataframes(zip_file_path)

    if final_df is not None:
        print("\n--- Processo concluído. DataFrame final criado. ---")
        print("Estrutura do DataFrame final:")
        final_df.info()
        print("Exemplo de 5 linhas:")
        print(final_df.head())

        output_path = "data/processed/base_consolidada.parquet"
        print(f"\nSalvando o DataFrame final em {output_path}...")
        final_df.to_parquet(output_path, engine='pyarrow')
        print("Arquivo salvo com sucesso.")