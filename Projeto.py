import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
from datetime import datetime
from joblib import dump

# 1. Conexão com o MySQL
usuario = 'root'
senha = 'Giojhow77*'
host = 'localhost'
porta = 3306
banco = 'telecom_churn'
engine = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{banco}')

# 2. Carregar dados
df = pd.read_sql('SELECT * FROM clientes_churn', engine)

# 3. Pré-processamento
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.fillna(0, inplace=True)

colunas_cat = df.select_dtypes(include='object').columns.tolist()
colunas_cat.remove('id_cliente')
colunas_cat.remove('cancelou')

for col in colunas_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df['cancelou'] = df['cancelou'].astype(str).str.strip().str.lower()
df['cancelou'] = df['cancelou'].map({'sim': 1, 'não': 0, 'nao': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0})
df = df.dropna(subset=['cancelou'])
df['cancelou'] = df['cancelou'].astype(int)

X = df.drop(['id_cliente', 'cancelou'], axis=1)
y = df['cancelou']

colunas_num = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[colunas_num] = scaler.fit_transform(X[colunas_num])

X.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Modelos
threshold = 0.5  # <<< USAR threshold balanceado

modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    resultados[nome] = {
        'modelo': modelo,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

for nome_modelo, metricas in resultados.items():
    print(f"\n{nome_modelo}")
    print(f"Acurácia: {metricas['accuracy']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"AUC-ROC: {metricas['roc_auc']:.4f}")
    print(f"Matriz de Confusão:\n{metricas['confusion_matrix']}")

melhor_modelo = max(resultados, key=lambda n: (resultados[n]['recall'], resultados[n]['roc_auc']))
melhor_m = resultados[melhor_modelo]['modelo']
print(f"\nMelhor modelo: {melhor_modelo}")


#Airflow
modelo_path = '/home/giovanni/projects/churn/airflow/models/modelo_churn.joblib'
scaler_path = '/home/giovanni/projects/churn/airflow/models/scaler.joblib'
dump(melhor_m, modelo_path)
dump(scaler, scaler_path)

# 5. Previsão final
X_full = df.drop(['cancelou'], axis=1)
df_final = X_full.copy()
df_final['prob_cancelamento'] = melhor_m.predict_proba(X_full.drop(['id_cliente'], axis=1))[:, 1]

# Segmentação (balanceada)
def segmentar(prob):
    if prob >= 0.6:
        return "Alto Risco"
    elif prob >= 0.3:
        return "Médio Risco"
    else:
        return "Baixo Risco"

df_final["segmento_cancelamento"] = df_final["prob_cancelamento"].apply(segmentar)
df_final["data_previsao"] = datetime.today().strftime('%Y-%m-%d')

# Diagnóstico
print("\nDistribuição dos segmentos de risco:")
print(df_final["segmento_cancelamento"].value_counts())

# 6. Salvar no banco
sql_create = """
CREATE TABLE IF NOT EXISTS churn_predictions_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_cliente VARCHAR(50),
    cobranca_mensal FLOAT,
    tempo_contrato INT,
    prob_cancelamento FLOAT,
    segmento_cancelamento VARCHAR(20),
    data_previsao DATE
);
"""
with engine.begin() as conn:
    conn.execute(text(sql_create))

cols = ["id_cliente", "cobranca_mensal", "tempo_contrato", "prob_cancelamento", "segmento_cancelamento", "data_previsao"]
payload = df_final[cols].to_dict(orient="records")

sql_insert = """
INSERT INTO churn_predictions_history
  (id_cliente, cobranca_mensal, tempo_contrato, prob_cancelamento, segmento_cancelamento, data_previsao)
VALUES
  (:id_cliente, :cobranca_mensal, :tempo_contrato, :prob_cancelamento, :segmento_cancelamento, :data_previsao);
"""
with engine.begin() as conn:
    conn.execute(text(sql_insert), payload)

print("\nTabela 'churn_predictions_history' atualizada com sucesso.")

# 7. Feature importance
if hasattr(melhor_m, "feature_importances_"):
    importances = melhor_m.feature_importances_
    feature_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    print("\nFeatures mais importantes:\n", feature_importance)
else:
    print("\nModelo escolhido não possui atributo 'feature_importances_' (ex: regressão logística).")
