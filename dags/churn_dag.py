from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import sys
import os

# Adiciona o caminho para os scripts
sys.path.insert(0, '/opt/airflow/scripts')

def run_churn_simple():
    """
    Função que executa o pipeline de churn
    """
    print("🚀 Iniciando pipeline de análise de churn...")
    
    try:
        from churn_pipeline import run_churn_pipeline
        print("✅ Módulo importado com sucesso!")
        
        result = run_churn_pipeline()
        print(f"✅ Pipeline executado. Resultado: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Erro no pipeline: {str(e)}")
        import traceback
        print(f"📋 Detalhes do erro: {traceback.format_exc()}")
        return False

def alerta_falha(context):
    """
    Função de callback para falhas
    """
    print(f"🚨 ALERTA: DAG falhou na task {context['task_instance'].task_id}")

# CONFIGURAÇÃO DA DAG AUTOMATIZADA
default_args = {
    'owner': 'giovanni',
    'start_date': datetime(2024, 1, 1),
    'retries': 2,  # Tentar novamente 2 vezes se falhar
    'retry_delay': timedelta(minutes=5),  # Esperar 5 minutos entre tentativas
    'email_on_failure': False,  # Altere para True se quiser emails
    'email_on_retry': False,
    'catchup': False  # Não executar runs passados
}


# (toda segunda-feira às 8h)
schedule_interval = '0 8 * * 1'


with DAG(
    'pipeline_churn_automatico',
    default_args=default_args,
    description='Pipeline automatizado de previsão de churn - Execução Diária',
    schedule_interval=schedule_interval,  # AGENDAMENTO AUTOMÁTICO
    catchup=False,
    tags=['churn', 'ml', 'automated'],
    on_failure_callback=alerta_falha,
    max_active_runs=1  
) as dag:

    executar_analise = PythonOperator(
        task_id='executar_analise_churn',
        python_callable=run_churn_simple,
        retries=2,
        retry_delay=timedelta(minutes=3)
    )

    # Enviar email de sucesso 
    email_sucesso = EmailOperator(
         task_id='enviar_email_sucesso',
         to='sousagiovanni@gmail.com',
         subject='Pipeline Churn - Executado com Sucesso',
         html_content='<h3>✅ Pipeline de Churn executado com sucesso!</h3><p>Novas previsões disponíveis no banco de dados.</p>'
     )

    # Definir dependências
    executar_analise  >> email_sucesso 

    executar_analise