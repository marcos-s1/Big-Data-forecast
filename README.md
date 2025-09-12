# Big-Data-forecast

Este projeto tem como objetivo realizar previsões utilizando técnicas de Big Data, com foco em análise de vendas semanais, agregação de dados e criação de features para modelos preditivos.

## Estrutura do Projeto

- **src/**: Scripts principais para carregamento, processamento e engenharia de dados.
- **notebooks/**: Jupyter Notebooks para exploração, análise e desenvolvimento dos modelos.
- **data/**: Dados brutos, processados e resultados das previsões.

## Principais Tecnologias

- Python (Pandas, NumPy, PySpark)
- Jupyter Notebook
- Parquet
- Big Data

## Como executar

1. Crie um ambiente virtual:
   ```
   python -m venv .venv
   ```
2. Ative o ambiente virtual:
   ```
   .venv\Scripts\activate
   ```
3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
4. Execute os scripts em `src/` ou os notebooks em `notebooks/`.

## Observações

- Para usar PySpark, é necessário instalar o Java (JDK) no sistema e configurar a variável de ambiente `JAVA_HOME`.
- Os dados devem ser colocados na pasta `data/raw/` conforme a estrutura esperada pelos scripts.

---
Projeto para estudos e experimentação com Big Data e previsão de vendas.

## Configuração do Java (JDK 21)

Este projeto foi testado e funciona com o **JDK 21**.  
Para utilizar o PySpark, siga os passos abaixo para instalar e configurar o Java no Windows:

1. **Baixe e instale o JDK 21**  
   - Acesse [Adoptium Temurin 21](https://adoptium.net/temurin/releases/?version=21) ou o site oficial da Oracle.
   - Instale o JDK normalmente.

2. **Configure as variáveis de ambiente**

   - Abra o Painel de Controle → Sistema → Configurações Avançadas → Variáveis de Ambiente.
   - Adicione ou edite a variável `JAVA_HOME` apontando para o diretório de instalação do JDK, por exemplo:
     ```
     C:\Program Files\Java\jdk-21.x.x
     ```
   - Edite a variável `Path` e adicione o caminho para a pasta `bin` do JDK:
     ```
     C:\Program Files\Java\jdk-21.x.x\bin
     ```

3. **Verifique a instalação**

   No terminal, execute:
   ```
   java -version
   echo %JAVA_HOME%
   ```

   O resultado deve mostrar a versão 21 do Java e o caminho correto do JDK.

4. **Reinicie o VS Code**  
   Para garantir que as variáveis de ambiente sejam reconhecidas.

---

Caso utilize outro sistema operacional, adapte os caminhos conforme necessário.