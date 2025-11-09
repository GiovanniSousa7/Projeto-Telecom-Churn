# 🚀 **Análise e Previsão de Churn no Setor de Telecomunicações**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/Apache_Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=power-bi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
</p>

---

## 🧠 **Sobre o Projeto**

Este projeto tem como objetivo **prever o risco de cancelamento (churn)** de clientes em uma empresa de telecomunicações, utilizando **Machine Learning**, **automação com Airflow** e **visualização interativa no Power BI**.

A iniciativa simula um pipeline completo de dados — desde o armazenamento no **MySQL**, passando pelo processamento e modelagem em **Python**, orquestração automatizada via **Apache Airflow**, até a análise final no **Power BI**.

O propósito é demonstrar como **dados históricos e aprendizado de máquina** podem ser usados para **antecipar o comportamento dos clientes**, otimizando estratégias de retenção e melhorando a tomada de decisão.

---

## ⚙️ **Principais Tecnologias Utilizadas**

| Etapa | Ferramenta | Descrição |
|-------|-------------|-----------|
| 🗄️ **Banco de Dados** | **MySQL** | Armazenamento e histórico dos clientes, base central para consumo de dados. |
| 🐍 **Modelagem** | **Python (Pandas, Scikit-learn, XGBoost)** | Limpeza, engenharia de atributos, treino e avaliação dos modelos de churn. |
| ⚙️ **Orquestração** | **Apache Airflow** | Automação e agendamento dos processos de modelagem e previsão. |
| 📊 **Visualização** | **Power BI** | Criação de dashboard interativo com KPIs e insights sobre churn. |
| 🐳 **Infraestrutura** | **Docker** | Contêinerização do ambiente para fácil replicação e execução. |

---

## 🧩 **Arquitetura do Projeto**

```mermaid
graph TD
    subgraph Data Flow
        A[🗄️ MySQL: Dados de Clientes e Histórico] --> B(🐍 Python: Processamento e Treinamento de Modelos);
        B --> C(⚙️ Airflow: Orquestração e Agendamento do Pipeline);
        C --> D[🔮 MySQL: Predições / Resultados Salvos];
        D --> E(📊 Power BI: Dashboards e KPIs Visuais);
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#cfc,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#ffc,stroke:#333,stroke-width:2px
```

### 🔄 **Fluxo do Processo**
1️⃣ **Coleta e Armazenamento** → Os dados são armazenados no **MySQL**, que serve como base central do projeto.  
2️⃣ **Modelagem e Predição** → Um script em **Python** faz a limpeza, engenharia de atributos e treinamento do modelo de *churn prediction*.  
3️⃣ **Automação com Airflow** → O **Apache Airflow** automatiza todo o fluxo de atualização e geração das novas predições.  
4️⃣ **Resultados e Visualização** → As predições são gravadas novamente no **MySQL**, e o **Power BI** consome esses dados em tempo real para exibir **KPIs e insights**.

> 💡 O fluxo pode ser resumido assim:  
> **MySQL → Python (Modelagem e Treinamento) → Airflow (Automação)**  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
> **Predições → Power BI (Visualização e KPIs)**  

---

## 📈 **Principais KPIs**

Os indicadores definidos no Power BI permitem avaliar a **saúde do negócio** e **detectar padrões de comportamento** de cancelamento:

| KPI | Descrição |
|------|------------|
| **Taxa de Churn (%)** | Percentual de clientes que cancelaram seus serviços. |
| **Risco Médio Por Serviço** | Percentual médio de cancelamento dos clientes por serviço. |
| **Clientes por Segmento de Risco** | Quantidade de clientes classificados como “Alto”, “Médio” e “Baixo” risco. |
| **Churn por Tipo de Contrato** | Distribuição de cancelamentos conforme o tipo de plano. |
| **Churn por Método de Pagamento** | Distribuição de cancelamentos conforme o método de pagamento. |

---

## 📊 **Dashboard Power BI**

O dashboard foi dividido em **4 páginas principais**:

1️⃣ **Dados Gerais** — KPIs globais e taxa total de churn.  
2️⃣ **Mapeamento de Perfil** — Características demográficas e contratuais.  
3️⃣ **Dados por Serviço** — Principais causas e padrões de churn por serviço.  
4️⃣ **Contratos e Pagamentos** — Análise de churn por contrato e método de pagamento.

> 🎨 O design segue um estilo **corporativo e minimalista**, com **layout intuitivo e visual limpo**, ideal para apresentações executivas.

---

## 🤖 **Modelos Utilizados**

Foram testados diferentes algoritmos de classificação binária:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**

O modelo final foi escolhido com base em **recall e AUC-ROC**, priorizando a **detecção correta dos clientes de alto risco de churn**.

---

## 🧠 **Principais Aprendizados**

- Aplicação prática de **Machine Learning** com dados de telecom.  
- Construção de **pipelines automatizados** via Airflow.  
- Criação de **dashboards corporativos** em Power BI.  
- Integração entre **banco de dados, modelagem e visualização**.  
- Uso do **Docker** para padronizar ambientes de execução.

---

## 🧱 **Estrutura do Projeto**

📦 projeto_churn/ <br>
├── dags/ → DAGs do Airflow <br>
├── scripts/ → Scripts Python de modelagem e predição <br>
├── models/ → Modelos e escalers salvos <br>
├── logs/ → Logs do Airflow <br>
├── Dockerfile → Imagem personalizada do Airflow <br>
├── docker-compose.yml → Orquestração dos contêineres <br>
├── clientes_churn_utf8.csv → Base de dados original <br>
├── churn_predictions_history.csv → Resultados do modelo <br>
└── README.md → Documentação do projeto


---

## 🧭 **Como Executar**

1️⃣ Clonar o repositório <br>
git clone https://github.com/GiovanniSousa7/projeto_churn.git<br>
cd projeto_churn

2️⃣ Iniciar o ambiente Docker<br>
docker compose up --build

3️⃣ Acessar o Airflow<br>
http://localhost:8080<br>
Usuário: admin<br>
Senha: admin

4️⃣ Visualizar o Dashboard<br>
Importe o arquivo .PBIX no Power BI Desktop.


👨🏻‍💻 Autor

Giovanni Sousa
📊 Data Science and Analytics | IA |  ETL | Python | SQL  | Power BI | Machine Learning

<p align="left"> <a href="https://www.linkedin.com/in/giovannisousap"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/> </a> <a href="https://github.com/GiovanniSousa7"> <img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white"/> </a> <a href="mailto:sousagiovanni19@gmail.com"> <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/> </a> </p>
