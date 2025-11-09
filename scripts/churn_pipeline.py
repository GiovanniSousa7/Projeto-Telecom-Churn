import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.utils import class_weight
from datetime import datetime
from joblib import dump
import os

def run_churn_pipeline():
    """
    Versão FINAL CORRIGIDA - Ajuste suave para manter três segmentos
    """
    print("🚀 Iniciando pipeline de churn CORRIGIDO...")
    
    # 1. Conexão com o MySQL 
    usuario = 'root'
    senha = 'Giojhow77*'
    host = 'host.docker.internal'  
    porta = 3306
    banco = 'telecom_churn'
    
    try:
        engine = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{banco}')
        print("✅ Conectado ao MySQL com sucesso!")
    except Exception as e:
        print(f"❌ Erro na conexão MySQL: {e}")
        return False

    # 2. Carregar dados
    try:
        df = pd.read_sql('SELECT * FROM clientes_churn', engine)
        print(f"📊 Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return False

    # 3. ANÁLISE INICIAL DOS DADOS
    print("\n🔍 ANALISANDO DISTRIBUIÇÃO DO TARGET:")
    if 'Churn' in df.columns:
        target_dist = df['Churn'].value_counts()
        print(f"Distribuição do Churn:\n{target_dist}")
        print(f"Taxa de Churn: {target_dist.get('Yes', 0) / len(df) * 100:.2f}%")
    elif 'churn' in df.columns:
        target_dist = df['churn'].value_counts()
        print(f"Distribuição do Churn:\n{target_dist}")

    # 4. PRÉ-PROCESSAMENTO CORRIGIDO
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    
    # Tratar coluna TotalCharges que pode ter valores vazios
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
    
    df.fillna(0, inplace=True)

    # Identificar colunas categóricas
    colunas_cat = df.select_dtypes(include='object').columns.tolist()
    
    # Colunas para remover (target + IDs)
    colunas_para_remover = []
    
    # Identificar coluna de ID
    coluna_id = None
    for id_col in ['customerID', 'customer_id', 'id_cliente']:
        if id_col in df.columns:
            coluna_id = id_col
            colunas_para_remover.append(id_col)
            if id_col in colunas_cat:
                colunas_cat.remove(id_col)
            break
    
    # Identificar coluna target
    coluna_target = None
    for target_col in ['Churn', 'churn', 'cancelou']:
        if target_col in df.columns:
            coluna_target = target_col
            colunas_para_remover.append(target_col)
            if target_col in colunas_cat:
                colunas_cat.remove(target_col)
            break

    if not coluna_target:
        print("❌ ERRO: Nenhuma coluna target encontrada!")
        return False

    print(f"🎯 Coluna target: {coluna_target}")
    print(f"🆔 Coluna ID: {coluna_id}")

    # Aplicar LabelEncoder CORRETAMENTE
    label_encoders = {}
    for col in colunas_cat:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"✅ Coluna '{col}' codificada")
        except Exception as e:
            print(f"⚠️ Erro ao codificar '{col}': {e}")
            df[col] = 0

    # Processar coluna target CORRETAMENTE
    df[coluna_target] = df[coluna_target].astype(str).str.strip().str.lower()
    
    # Mapeamento para inglês
    mapeamento_target = {
        'yes': 1, 'no': 0, 'sim': 1, 'não': 0, 'nao': 0,
        '1': 1, '0': 0, 'true': 1, 'false': 0,
        'churn': 1, 'no churn': 0
    }
    
    df[coluna_target] = df[coluna_target].map(mapeamento_target)
    
    # Verificar se o mapeamento funcionou
    target_after_map = df[coluna_target].value_counts()
    print(f"🎯 Distribuição após mapeamento:\n{target_after_map}")
    
    if df[coluna_target].isna().any():
        print("⚠️ Valores não mapeados encontrados no target. Preenchendo com 0.")
        df[coluna_target].fillna(0, inplace=True)
    
    df[coluna_target] = df[coluna_target].astype(int)

    # Preparar features e target
    X = df.drop(colunas_para_remover, axis=1)
    y = df[coluna_target]

    print(f"🔢 Shape final - X: {X.shape}, y: {y.shape}")
    print(f"🎯 Balanceamento - 0: {(y == 0).sum()}, 1: {(y == 1).sum()}")

    # 5. NORMALIZAÇÃO APENAS DAS NUMÉRICAS
    colunas_num = X.select_dtypes(include=['int64', 'float64']).columns
    # Remover colunas que são categóricas codificadas (valores únicos limitados)
    colunas_para_normalizar = []
    for col in colunas_num:
        if X[col].nunique() > 10:  # Apenas colunas com muitos valores únicos
            colunas_para_normalizar.append(col)
    
    if len(colunas_para_normalizar) > 0:
        scaler = StandardScaler()
        X[colunas_para_normalizar] = scaler.fit_transform(X[colunas_para_normalizar])
        print(f"✅ {len(colunas_para_normalizar)} colunas normalizadas")
    else:
        scaler = None
        print("⚠️ Nenhuma coluna para normalizar")

    # Limpeza final
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # 6. SPLIT E BALANCEAMENTO
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Split - Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"🎯 Balanceamento Treino - 0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()}")

    # 7. MODELOS COM BALANCEAMENTO CORRIGIDO
    # Calcular pesos para balanceamento
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f"⚖️ Pesos de classe: {class_weight_dict}")

    modelos = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42
        )
    }

    resultados = {}
    
    # THRESHOLD MAIS CONSERVADOR - REDUZIR FALSOS POSITIVOS
    threshold_otimizado = 0.8  # Aumentei de 0.7 para 0.8 (mais conservador)
    
    for nome, modelo in modelos.items():
        print(f"🤖 Treinando {nome}...")
        modelo.fit(X_train, y_train)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        # Usar threshold otimizado
        y_pred = (y_proba >= threshold_otimizado).astype(int)

        resultados[nome] = {
            'modelo': modelo,
            'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"   📊 Probs - Min: {y_proba.min():.3f}, Max: {y_proba.max():.3f}, Avg: {y_proba.mean():.3f}")

    # Exibir resultados detalhados
    for nome_modelo, metricas in resultados.items():
        print(f"\n📈 {nome_modelo} (Threshold: {threshold_otimizado})")
        print(f"   Acurácia: {metricas['accuracy']:.4f}")
        print(f"   Recall: {metricas['recall']:.4f}")
        print(f"   AUC-ROC: {metricas['roc_auc']:.4f}")
        print(f"   Matriz Confusão:\n{metricas['confusion_matrix']}")

    # Selecionar melhor modelo baseado em AUC-ROC
    melhor_modelo = max(resultados, key=lambda n: resultados[n]['roc_auc'])
    melhor_m = resultados[melhor_modelo]['modelo']
    print(f"\n🏆 Melhor modelo: {melhor_modelo}")

    # 8. SALVAR MODELOS
    modelo_path = '/opt/airflow/models/modelo_churn.joblib'
    scaler_path = '/opt/airflow/models/scaler.joblib'
    
    os.makedirs('/opt/airflow/models', exist_ok=True)
    
    dump(melhor_m, modelo_path)
    if scaler:
        dump(scaler, scaler_path)
    print(f"💾 Modelos salvos")

    # 9. PREVISÃO FINAL CORRIGIDA - COM THRESHOLD APLICADO SUAVEMENTE
    print("\n🎯 PREPARANDO PREVISÕES FINAIS...")
    
    # Carregar dados ORIGINAIS novamente para previsão
    df_original = pd.read_sql('SELECT * FROM clientes_churn', engine)
    
    # Manter uma cópia dos dados originais para o resultado final
    df_final = df_original.copy()
    
    # Pré-processar APENAS para previsão (igual ao treino)
    df_previsao = df_original.copy()
    
    # Aplicar o MESMO pré-processamento do treino
    df_previsao.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    
    if 'TotalCharges' in df_previsao.columns:
        df_previsao['TotalCharges'] = pd.to_numeric(df_previsao['TotalCharges'], errors='coerce')
        df_previsao['TotalCharges'].fillna(0, inplace=True)
    
    df_previsao.fillna(0, inplace=True)
    
    # Aplicar MESMO LabelEncoder do treino
    for col in colunas_cat:
        if col in df_previsao.columns:
            try:
                le = label_encoders[col]  # Usar o MESMO encoder do treino
                df_previsao[col] = le.transform(df_previsao[col].astype(str))
            except Exception as e:
                print(f"⚠️ Erro ao transformar '{col}': {e}")
                df_previsao[col] = 0
    
    # Aplicar MESMA normalização do treino
    X_previsao = df_previsao.drop(colunas_para_remover, axis=1)
    
    if scaler and len(colunas_para_normalizar) > 0:
        X_previsao[colunas_para_normalizar] = scaler.transform(X_previsao[colunas_para_normalizar])
    
    X_previsao.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_previsao.fillna(0, inplace=True)
    
    # AGORA SIM: Fazer previsões corretas COM THRESHOLD APLICADO SUAVEMENTE
    print("🤖 Fazendo previsões com o modelo treinado...")
    probas_brutas = melhor_m.predict_proba(X_previsao)[:, 1]
    
    # ✅ APLICAR THRESHOLD NAS PREVISÕES FINAIS - VERSÃO MAIS SUAVE
    print(f"🎯 Aplicando threshold de {threshold_otimizado} nas previsões finais...")
    
    # Estratégia mais suave: Redução progressiva mantendo três segmentos
    def ajuste_suave(prob, threshold=0.8):
        if prob >= threshold:
            return prob  # Mantém alto risco (acima de 0.8)
        elif prob >= 0.5:  # Médio risco superior
            return prob * 0.85  # Redução muito leve de 15%
        elif prob >= 0.4:  # Médio risco inferior  
            return prob * 0.75  # Redução leve de 25%
        else:  # Baixo risco
            return prob * 0.6  # Redução moderada de 40%
    
    # Aplicar ajuste suave a todas as probabilidades
    probas_ajustadas = np.array([ajuste_suave(p, threshold_otimizado) for p in probas_brutas])
    
    df_final['prob_cancelamento'] = probas_ajustadas

    print(f"\n📊 COMPARAÇÃO PROBABILIDADES:")
    print(f"   Brutas  - Min: {probas_brutas.min():.4f}, Max: {probas_brutas.max():.4f}, Avg: {probas_brutas.mean():.4f}")
    print(f"   Ajustadas - Min: {probas_ajustadas.min():.4f}, Max: {probas_ajustadas.max():.4f}, Avg: {probas_ajustadas.mean():.4f}")
    print(f"   Redução média: {(probas_brutas.mean() - probas_ajustadas.mean()) / probas_brutas.mean() * 100:.1f}%")

    # ✅ SEGMENTAÇÃO CORRIGIDA - THRESHOLDS MAIS REALISTAS
    def segmentar_corrigido(prob):
        if prob >= 0.7:    # Alto Risco - apenas os realmente críticos
            return "Alto Risco"
        elif prob >= 0.4:  # Médio Risco
            return "Médio Risco"
        else:              # Baixo Risco
            return "Baixo Risco"

    df_final["segmento_cancelamento"] = df_final["prob_cancelamento"].apply(segmentar_corrigido)
    df_final["data_previsao"] = datetime.today().strftime('%Y-%m-%d')

    print("\n📊 DISTRIBUIÇÃO REAL DOS SEGMENTOS:")
    distribuicao = df_final["segmento_cancelamento"].value_counts()
    print(distribuicao)
    
    for segmento, count in distribuicao.items():
        percentual = (count / len(df_final)) * 100
        print(f"   {segmento}: {count} clientes ({percentual:.1f}%)")

    # 10. SALVAR NO BANCO - VERSÃO CORRIGIDA COM DEBUG DETALHADO
    try:
        print("\n🔄 INICIANDO SALVAMENTO NO BANCO...")
        
        # DEBUG 1: Verificar estado atual da tabela
        with engine.begin() as conn:
            table_exists = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'telecom_churn' 
                AND table_name = 'churn_predictions_history'
            """)).scalar()
            
            current_count = conn.execute(text("SELECT COUNT(*) FROM churn_predictions_history")).scalar()
            
            print(f"📋 Tabela existe? {table_exists > 0}")
            print(f"📊 Registros atuais na tabela: {current_count}")

        # DEBUG 2: Verificar dados antes de salvar
        print(f"🎯 Dados para salvar: {len(df_final)} registros")
        print(f"📈 Segmentos a serem salvos: {df_final['segmento_cancelamento'].value_counts().to_dict()}")
        
        # 1. LIMPAR PREVISÕES ANTIGAS COM CONFIRMAÇÃO
        print("🗑️ Limpando registros antigos...")
        with engine.begin() as conn:
            deleted_count = conn.execute(text("DELETE FROM churn_predictions_history")).rowcount
            print(f"✅ Registros antigos removidos: {deleted_count}")

        # 2. PREPARAR DADOS NOVOS
        mapeamento_colunas = {
            'customerID': 'id_cliente',
            'MonthlyCharges': 'cobranca_mensal', 
            'tenure': 'tempo_contrato'
        }
        
        colunas_para_salvar = []
        for col in ['customerID', 'MonthlyCharges', 'tenure']:
            if col in df_final.columns:
                colunas_para_salvar.append(col)
        
        colunas_para_salvar.extend(['prob_cancelamento', 'segmento_cancelamento', 'data_previsao'])
        
        df_para_salvar = df_final[colunas_para_salvar].copy()
        df_para_salvar = df_para_salvar.rename(columns=mapeamento_colunas)
        
        # DEBUG 3: Verificar dados transformados
        print(f"📦 Dados após transformação: {len(df_para_salvar)} registros")
        print(f"📝 Colunas finais: {df_para_salvar.columns.tolist()}")
        print(f"🔍 Amostra dos dados:")
        print(df_para_salvar[['id_cliente', 'prob_cancelamento', 'segmento_cancelamento']].head(3))
        
        payload = df_para_salvar.to_dict(orient="records")
        
        # 3. INSERIR NOVOS DADOS
        print("💾 Inserindo novos registros...")
        sql_insert = """
        INSERT INTO churn_predictions_history
          (id_cliente, cobranca_mensal, tempo_contrato, prob_cancelamento, segmento_cancelamento, data_previsao)
        VALUES
          (:id_cliente, :cobranca_mensal, :tempo_contrato, :prob_cancelamento, :segmento_cancelamento, :data_previsao);
        """
        
        with engine.begin() as conn:
            result = conn.execute(text(sql_insert), payload)
            inserted_count = result.rowcount if result.rowcount != -1 else len(payload)
            print(f"✅ Novos registros inseridos: {inserted_count}")

        # 4. VERIFICAR SE OS DADOS REALMENTE FORAM SALVOS
        print("🔍 Verificando salvamento...")
        with engine.begin() as conn:
            saved_count = conn.execute(text("SELECT COUNT(*) FROM churn_predictions_history")).scalar()
            print(f"📊 Total de registros na tabela AGORA: {saved_count}")
            
            # Verificar segmentos salvos
            segmentos_salvos = conn.execute(text("""
                SELECT segmento_cancelamento, COUNT(*) 
                FROM churn_predictions_history 
                GROUP BY segmento_cancelamento
            """)).fetchall()
            
            print("🎯 Distribuição final no banco:")
            for segmento, count in segmentos_salvos:
                percentual = (count / saved_count) * 100
                print(f"   {segmento}: {count} registros ({percentual:.1f}%)")

        # VERIFICAÇÃO FINAL
        if saved_count == len(df_final):
            print(f"🎉 SUCESSO! Todos os {saved_count} registros foram salvos corretamente!")
        else:
            print(f"⚠️ AVISO: Esperados {len(df_final)} registros, mas temos {saved_count} no banco")
            
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao salvar no banco: {e}")
        import traceback
        print(f"📋 DETALHES COMPLETOS DO ERRO:")
        print(traceback.format_exc())
        return False

    # 11. RELATÓRIO FINAL
    print("\n🎉 PIPELINE CORRIGIDO EXECUTADO COM SUCESSO!")
    print("🔧 Principais correções aplicadas:")
    print("   ✅ Balanceamento de classes com pesos")
    print("   ✅ Threshold conservador (0.8) para reduzir falsos positivos")
    print("   ✅ Threshold APLICADO suavemente nas previsões finais")
    print("   ✅ Manutenção dos três segmentos (Alto, Médio, Baixo Risco)")
    print("   ✅ Previsões corretas nos dados originais")
    print("   ✅ Segmentação realista baseada em probabilidades ajustadas")

    return True

if __name__ == "__main__":
    run_churn_pipeline()