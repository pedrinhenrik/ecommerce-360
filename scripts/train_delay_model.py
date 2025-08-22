"""
Treina um classificador de risco de atraso (>0 dias) usando features geradas em build_features.py.
Saídas:
- models/delay_xgb.pkl
- models/num_scaler.pkl
- data/processed/predicted_delay.csv
- docs/metrics_delay_model.json
- docs/plots/{roc.png, pr.png, confusion.png, shap_global.png, shap_example.png}
"""

from __future__ import annotations
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from xgboost import XGBClassifier
import joblib
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"
for p in [MODELS, DOCS, PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

print("🚀 Iniciando treinamento do modelo de risco de atraso")

# ── Carregamento das features (Parquet → CSV fallback)
df_path_parquet = PROCESSED / "dataset_delay_features.parquet"
df_path_csv = PROCESSED / "dataset_delay_features.csv"
if df_path_parquet.exists():
    df = pd.read_parquet(df_path_parquet)
    print(f"📥 Dataset carregado (Parquet): {df_path_parquet}")
elif df_path_csv.exists():
    df = pd.read_csv(df_path_csv, parse_dates=["order_purchase_timestamp"])
    print(f"📥 Dataset carregado (CSV): {df_path_csv}")
else:
    raise FileNotFoundError("❌ Não encontrei dataset de features. Rode antes: scripts/build_features.py")

# ── Definições de colunas
TARGET = "y_atraso"
NUM = ["dias_previstos_entrega", "dist_km", "peso_kg", "vol_m3", "densidade", "frete_ratio", "ticket", "mes_num"]
CAT = ["seller_state", "customer_state", "product_category_name"]

# Garantias de tipo
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in CAT:
    df[c] = df[c].astype(str).fillna("missing")

# ── Split temporal: treino (anos < max) vs teste (ano = max)
df["ano"] = df["order_purchase_timestamp"].dt.year
max_year = int(df["ano"].max())
train = df[df["ano"] < max_year].copy()
test = df[df["ano"] == max_year].copy()
if len(test) == 0:  # fallback se dataset for de um único ano
    split_point = int(0.8 * len(df))
    train, test = df.iloc[:split_point].copy(), df.iloc[split_point:].copy()
    print("⚠️ Dataset com único ano — usando split 80/20 por ordem temporal")

print(f"🧪 Split temporal → treino: {len(train):,} | teste: {len(test):,} | ano_teste: {max_year}")

X_train, y_train = train[NUM + CAT].copy(), train[TARGET].astype(int)
X_test, y_test = test[NUM + CAT].copy(), test[TARGET].astype(int)

# ── Codificação categórica: ordinal estável (vistas no treino; desconhecidas = -1)
encoders = {}
for c in CAT:
    mapping = {k: i for i, k in enumerate(sorted(X_train[c].astype(str).unique()))}
    encoders[c] = mapping
    X_train[c] = X_train[c].map(mapping).fillna(-1).astype("int32")
    X_test[c]  = X_test[c].map(mapping).fillna(-1).astype("int32")

# ── Escala numérica (estabiliza booster e gradientes)
scaler = StandardScaler()
X_train[NUM] = scaler.fit_transform(X_train[NUM])
X_test[NUM]  = scaler.transform(X_test[NUM])

# ── Modelo (hiperparâmetros conservadores e reprodutíveis)
clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    n_jobs=-1,
    random_state=42
)

print("🔧 Treinando XGBoost...")
clf.fit(X_train, y_train)
print("✅ Treinamento concluído")

# ── Avaliação probabilística (threshold-agnostic)
proba_test = clf.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, proba_test)
ap = average_precision_score(y_test, proba_test)
print(f"📊 Métricas (teste) → ROC AUC: {roc:.3f} | AP: {ap:.3f}")

# ── Curvas ROC e PR
fpr, tpr, _ = roc_curve(y_test, proba_test)
prec, rec, _ = precision_recall_curve(y_test, proba_test)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Delay Risk")
plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(PLOTS / "roc.png", dpi=150)

plt.figure()
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall — Delay Risk")
plt.legend(loc="lower left")
plt.tight_layout(); plt.savefig(PLOTS / "pr.png", dpi=150)

# ── Matriz de confusão para um threshold operacional (ajustável)
threshold = 0.50
y_pred = (proba_test >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["No-Delay", "Delay"])
disp.plot(values_format="d")
plt.title(f"Confusion Matrix — threshold {threshold:.2f}")
plt.tight_layout(); plt.savefig(PLOTS / "confusion.png", dpi=150)

report = classification_report(y_test, y_pred, output_dict=True)

# ── SHAP: explicabilidade global + exemplo individual
print("🧠 Calculando SHAP (amostra controlada para performance)...")
explainer = shap.TreeExplainer(clf)
sample_n = min(2000, len(X_test))
sample_idx = np.random.RandomState(42).choice(len(X_test), size=sample_n, replace=False)
X_test_sample = X_test.iloc[sample_idx]

shap_values = explainer.shap_values(X_test_sample)
plt.figure()
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig(PLOTS / "shap_global.png", dpi=150); plt.close()

# Caso individual com maior probabilidade prevista de atraso
top_idx = int(np.argmax(proba_test))
row = X_test.iloc[[top_idx]]
sv_row = explainer.shap_values(row)
plt.figure()
shap.waterfall_plot(
    shap.Explanation(values=sv_row[0],
                     base_values=explainer.expected_value,
                     data=row.iloc[0].values,
                     feature_names=row.columns.tolist()),
    show=False
)
plt.tight_layout(); plt.savefig(PLOTS / "shap_example.png", dpi=150); plt.close()
print("✅ SHAP gerado")

# ── Persistência: modelo, scaler e métricas
joblib.dump(clf, MODELS / "delay_xgb.pkl")
joblib.dump(scaler, MODELS / "num_scaler.pkl")

metrics = {
    "roc_auc": float(roc),
    "average_precision": float(ap),
    "threshold": threshold,
    "support_test": int(len(y_test)),
    "classification_report": report
}
with open(DOCS / "metrics_delay_model.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"💾 Artefatos salvos em {MODELS} e métricas em {DOCS/'metrics_delay_model.json'}")

# ── Saída para Tableau: probabilidade por order_id (marca split)
pred_test = test[["order_id"]].copy()
pred_test["prob_atraso"] = proba_test
pred_test["y_true"] = y_test.values
pred_test["split"] = "test"

# (opcional) salvar também probabilidades de treino para análises internas
proba_train = clf.predict_proba(X_train)[:, 1]
pred_train = train[["order_id"]].copy()
pred_train["prob_atraso"] = proba_train
pred_train["y_true"] = y_train.values
pred_train["split"] = "train"

pred_all = pd.concat([pred_train, pred_test], ignore_index=True)
pred_all.to_csv(PROCESSED / "predicted_delay.csv", index=False)

print(f"✅ Predições salvas em {PROCESSED/'predicted_delay.csv'}")
print("🎉 Treinamento finalizado com sucesso!")