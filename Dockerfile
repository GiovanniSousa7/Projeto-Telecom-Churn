# Usa imagem oficial do Airflow (versão que você escolheu)
FROM apache/airflow:2.7.0

# Troca para root para instalar pacotes system-wide
USER root

# Atualiza pip e instala pacotes no site-packages global.
# (Sem --user; assim todos os processos no container conseguem importar)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      pandas \
      scikit-learn \
      xgboost \
      sqlalchemy \
      pymysql \
      joblib

# Volta para o usuário airflow (boa prática do upstream)
USER airflow
