### **Big-Data-forecast ✨**
---

Este projeto tem como objetivo realizar previsões de vendas semanais, utilizando técnicas de Big Data, com foco em agregação de dados, engenharia de features e modelagem preditiva.

---

### **Estrutura do Projeto 📂**

* **`src/`**: Contém o código principal do nosso pipeline, dividido em scripts para cada etapa do processo.
* **`notebooks/`**: Notebooks Jupyter para análises exploratórias, testes e visualizações dos resultados.
* **`data/raw/`**: Onde os dados brutos (como o arquivo `.zip`) devem ser armazenados.
* **`data/processed/`**: Armazena os dados intermediários e a base final com features, prontos para a modelagem.
* **`models/`**: Pasta para salvar o modelo de previsão treinado e os arquivos de configuração (como a lista de features selecionadas).
* **`requirements.txt`**: Lista todas as bibliotecas necessárias para o projeto.
* **`README.md`**: Este arquivo, que serve como guia para o projeto.

---

### **O Pipeline do Projeto 🚀**

Nosso projeto segue um fluxo de trabalho estruturado para garantir a melhor performance e precisão:

1.  **Carregar e Juntar os Dados**: Lemos os arquivos `Parquet` de dentro do arquivo `.zip` e os unimos em uma única base de dados consolidada.
2.  **Engenharia de Features**: Criamos novas variáveis poderosas a partir dos dados brutos, como o ranque de semanas (`week_rank`), a semana do mês (`week_of_month`) e features defasadas (`lagged features`) que olham para o passado para prever o futuro.
3.  **Seleção de Features**: Utilizamos uma abordagem combinada para encontrar as melhores variáveis:
    * **Triagem Rápida**: Removemos features com pouca importância no modelo.
    * **Backward Selection**: Para as features restantes, removemos uma por uma e avaliamos o impacto no WMAPE, garantindo que a remoção de uma variável não prejudique o modelo.
4.  **Treinamento do Modelo Final**: Treinamos o modelo CatBoost com a base de dados completa de 2022, usando apenas as features que selecionamos.
5.  **Previsão e Scoragem**: Geramos as previsões para janeiro de 2023, semana a semana, usando o modelo treinado.

---

### **O Modelo Escolhido: CatBoost 🎯**

Para este projeto, utilizamos o **CatBoost**, um modelo de `gradient boosting` de última geração.

**Vantagens do CatBoost:**
* **Performance e GPU**: O CatBoost é extremamente rápido, especialmente com a aceleração de **GPU**, tornando o treinamento mais eficiente.
* **Categorias Nativas**: Ele lida com variáveis categóricas de forma nativa e otimizada, o que nos poupa do pré-processamento manual e de problemas como a alta dimensionalidade.
* **Robustez**: Ele é projetado para evitar `overfitting`, sendo ideal para o nosso projeto.

---

### **Configuração do Ambiente ☁️**

Este projeto foi desenvolvido e testado no **Google Colab**, pois sua integração com o PySpark e acesso a GPUs é facilitada.

Para uma execução sem problemas e com o máximo de performance, recomendamos a seguinte configuração de hardware:

* **GPU**: Uma **NVIDIA Tesla A100** para acelerar o treinamento do CatBoost.
* **Memória RAM**: **64GB ou mais** para lidar com o volume total de dados e evitar erros de `OutOfMemoryError` durante as operações mais pesadas do PySpark.

---

### **Como executar ⚙️**

1.  Abra o seu notebook no Google Colab.
2.  Monte o seu Google Drive para que o notebook possa acessar os dados e salvar os resultados.
3.  Instale as bibliotecas necessárias usando o `requirements.txt`.
4.  Execute as células do notebook em sequência, seguindo a lógica do pipeline.
