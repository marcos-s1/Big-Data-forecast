# Big-Data-forecast ✨

Este projeto tem como objetivo realizar previsões utilizando técnicas de Big Data, com foco em análise de vendas semanais, agregação de dados e criação de features para modelos preditivos.

---

## Estrutura do Projeto 📂

- **src/**: Scripts principais para carregamento, processamento e engenharia de dados.
- **notebooks/**: Jupyter Notebooks para exploração, análise e desenvolvimento dos modelos.
- **data/**: Dados brutos, processados e resultados das previsões.

---

## Principais Tecnologias 🛠️

- Python (Pandas, NumPy, PySpark)
- Jupyter Notebook
- Parquet
- Big Data

---

## Como executar ⚙️

1.  Crie um ambiente virtual:
    ```bash
    python -m venv .venv
    ```
2.  Ative o ambiente virtual:
    ```bash
    .venv\Scripts\activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4.  Execute os scripts em `src/` ou os notebooks em `notebooks/`.

---

## Configuração do Ambiente 💻

Este projeto foi desenvolvido e testado no **VS Code** e requer a configuração correta do Java e do Python para o PySpark funcionar.

### 1. Instalação e Configuração do Java (JDK 17) ☕

O Apache Spark 3.5.1 é compatível com o **JDK 17**. Para evitar erros, siga estas instruções:

1.  **Baixe e instale o JDK 17:**
    - Acesse [Adoptium Temurin 17](https://adoptium.net/temurin/releases/?version=17) ou o site oficial da Oracle.
    - Instale o JDK normalmente.

2.  **Configure as variáveis de ambiente:**
    - No Windows, pesquise por "Variáveis de Ambiente do Sistema".
    - Na janela que se abrir, clique em **"Variáveis de Ambiente..."**.
    - Adicione ou edite a variável de sistema `JAVA_HOME` apontando para o diretório de instalação do JDK 17.

3.  **Verifique a instalação:**
    No terminal, execute:
    ```bash
    java -version
    echo %JAVA_HOME%
    ```
    O resultado deve mostrar a versão 17 do Java e o caminho correto do JDK.

### 2. Configuração do Ambiente Python 🐍

Para garantir que o PySpark use o ambiente virtual do seu projeto, você precisa definir a variável de ambiente `PYSPARK_PYTHON` para o caminho correto do `python.exe` dentro do seu `.venv`.

**Opção recomendada:** Defina a variável permanentemente, seguindo o mesmo processo do `JAVA_HOME`.

- **Nome da variável:** `PYSPARK_PYTHON`
- **Valor da variável:** O caminho completo para o `python.exe` dentro da sua pasta `.venv`.

**Exemplo:**

- **No Windows:**
`C:\Users\SeuUsuario\Caminho\Para\O\Projeto\.venv\Scripts\python.exe`

- **No macOS/Linux:**
`/home/seu-usuario/caminho/para/o/projeto/.venv/bin/python`

### 3. Configuração do Hadoop (Apenas para Windows) ⚙️

Este passo é crucial para usuários de Windows, pois o PySpark depende de um ambiente Hadoop local para operações de arquivos.

1.  **Baixe os Arquivos Essenciais:**
    - Baixe os arquivos `winutils.exe` e `hadoop.dll` para a versão do Hadoop compatível com o seu Spark.
    - Um repositório comum para isso é: `https://github.com/steveloughran/winutils/`

2.  **Crie a Pasta do Hadoop:**
    - Crie uma pasta `hadoop` na raiz do seu disco `C:` (ex: `C:\hadoop`).
    - Dentro dela, crie uma subpasta `bin` (ex: `C:\hadoop\bin`).

3.  **Mova os Arquivos:**
    - Mova `winutils.exe` e `hadoop.dll` para a pasta `C:\hadoop\bin`.

4.  **Configure a Variável de Ambiente `HADOOP_HOME`:**
    - Siga o mesmo processo de "Variáveis de Ambiente" usado para o `JAVA_HOME`.
    - Crie uma nova variável de sistema:
        - **Nome:** `HADOOP_HOME`
        - **Valor:** `C:\hadoop`
    - Edite a variável de sistema `Path` e adicione o caminho `C:\hadoop\bin`.

5.  **Verifique a configuração:**
    No terminal, reinicie o ambiente e execute:
    ```bash
    echo %HADOOP_HOME%
    ```
    O resultado deve ser `C:\hadoop`.

---

## Observações Importantes 📌

- Os dados devem ser colocados na pasta `data/raw/` conforme a estrutura esperada pelos scripts.
- O arquivo `.gitignore` deve incluir a pasta `data/` para evitar o upload de arquivos grandes para o repositório.

---

Projeto para estudos e experimentação com Big Data e previsão de vendas. 📈