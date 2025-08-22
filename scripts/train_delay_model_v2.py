"""
Treino v2 do classificador de risco de atraso (>0 dias) com melhorias operacionais.
Saídas principais:
- models/delay_xgb.pkl  [ou models/delay_xgb_calibrated.pkl]
- models/num_scaler.pkl
- data/processed/predicted_delay.csv
- docs/metrics_delay_model.json
- docs/plots/{roc.png, pr.png, confusion.png, shap_global.png, shap_example.png}
- docs/decile_lift.csv
- docs/segment_metrics_customer_state.csv
- docs/segment_metrics_category.csv
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
    classification_report, fbeta_score
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# ==== Configuração de execução ====
USE_CALIBRATION = True            # calibração isotônica das probabilidades
FBETA = 2.0                       # prioriza recall
N_SHAP_SAMPLE = 2000              # amostra para SHAP
RANDOM_STATE = 42

# ==== Paths ====
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"
for p in [MODELS, DOCS, PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

print("🚀 Iniciando treinamento v2 do modelo de risco de atraso")

# ==== Carregamento das features (Parquet → CSV fallback) ====
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

# ==== Definições de colunas ====
TARGET = "y_atraso"
NUM = ["dias_previstos_entrega", "dist_km", "peso_kg", "vol_m3", "densidade", "frete_ratio", "ticket", "mes_num"]
CAT = ["seller_state", "customer_state", "product_category_name"]

# Garantias de tipo
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
if "mes_num" not in df.columns:
    df["mes_num"] = df["order_purchase_timestamp"].dt.month.astype("int8")
for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in CAT:
    df[c] = df[c].astype(str).fillna("missing")

# ==== Split temporal: treino (< ano máximo) vs teste (ano máximo) ====
df["ano"] = df["order_purchase_timestamp"].dt.year
max_year = int(df["ano"].max())
train = df[df["ano"] < max_year].copy()
test = df[df["ano"] == max_year].copy()
if len(test) == 0:  # fallback para datasets de um único ano
    split_point = int(0.8 * len(df))
    train, test = df.iloc[:split_point].copy(), df.iloc[split_point:].copy()
    print("⚠️ Dataset com único ano. Usando split 80/20 por ordem temporal")

print(f"🧪 Split temporal → treino: {len(train):,} | teste: {len(test):,} | ano_teste: {max_year}")

X_train, y_train = train[NUM + CAT].copy(), train[TARGET].astype(int)
X_test,  y_test  = test[NUM + CAT].copy(),  test[TARGET].astype(int)

# ==== Codificação categórica: ordinal estável ====
encoders = {}
for c in CAT:
    mapping = {k: i for i, k in enumerate(sorted(X_train[c].astype(str).unique()))}
    encoders[c] = mapping
    X_train[c] = X_train[c].map(mapping).fillna(-1).astype("int32")
    X_test[c]  = X_test[c].map(mapping).fillna(-1).astype("int32")

# ==== Escala numérica ====
scaler = StandardScaler()
X_train[NUM] = scaler.fit_transform(X_train[NUM])
X_test[NUM]  = scaler.transform(X_test[NUM])

# ==== Modelo base ====
ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=ratio,   # sensível ao desbalanceamento
    objective="binary:logistic",
    eval_metric="auc",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("🔧 Treinando XGBoost...")
clf.fit(X_train, y_train)
print("✅ Treinamento concluído")

# ==== Calibração opcional das probabilidades ====
if USE_CALIBRATION:
    print("🔄 Calibrando probabilidades (isotônico, cv=3)...")
    cal = CalibratedClassifierCV(clf, method="isotonic", cv=3, n_jobs=-1)
    cal.fit(X_train, y_train)
    model_for_proba = cal
    model_path = MODELS / "delay_xgb_calibrated.pkl"
    print("✅ Calibração concluída")
else:
    model_for_proba = clf
    model_path = MODELS / "delay_xgb.pkl"

# ==== Avaliação threshold-agnostic (probabilística) ====
proba_test = model_for_proba.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, proba_test)
ap = average_precision_score(y_test, proba_test)
print(f"📊 Métricas (teste) → ROC AUC: {roc:.3f} | AP: {ap:.3f}")

# Curvas ROC e PR
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

# ==== Busca de threshold ótimo ====
def search_best_threshold(y_true: np.ndarray, proba: np.ndarray, beta: float = 2.0):
    grid = np.linspace(0.05, 0.95, 181)
    best = {"t": 0.5, "fbeta": -1.0, "youden": -1.0, "precision": 0.0, "recall": 0.0}
    for t in grid:
        y_pred = (proba >= t).astype(int)
        # métricas básicas
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        # F-beta
        fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        # Youden J
        tpr_val = rec
        fpr_val = fp / max(1, (fp + tn))
        youden = tpr_val - fpr_val
        # regra: prioriza F-beta. em empate, maior recall. se empatar, maior Youden.
        is_better = (
            (fbeta > best["fbeta"]) or
            (np.isclose(fbeta, best["fbeta"]) and rec > best["recall"]) or
            (np.isclose(fbeta, best["fbeta"]) and np.isclose(rec, best["recall"]) and youden > best["youden"])
        )
        if is_better:
            best = {"t": float(t), "fbeta": float(fbeta), "youden": float(youden),
                    "precision": float(prec), "recall": float(rec)}
    return best

print("🎯 Buscando threshold ótimo por F-beta e Youden...")
best = search_best_threshold(y_test.values, proba_test, beta=FBETA)
threshold = best["t"]
print(f"✅ Threshold selecionado: {threshold:.2f} | F{FBETA:.0f}={best['fbeta']:.3f} | P={best['precision']:.3f} | R={best['recall']:.3f}")

# Matriz de confusão no threshold escolhido
y_pred = (proba_test >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["No-Delay", "Delay"])
disp.plot(values_format="d")
plt.title(f"Confusion Matrix — threshold {threshold:.2f}")
plt.tight_layout(); plt.savefig(PLOTS / "confusion.png", dpi=150)

report = classification_report(y_test, y_pred, output_dict=True)

# ==== SHAP: explicabilidade ====
print("🧠 Calculando SHAP (amostra controlada para performance)...")
# usa o modelo base para o explainer quando calibrado
explainer_model = clf if USE_CALIBRATION else clf
explainer = shap.TreeExplainer(explainer_model)
sample_n = min(N_SHAP_SAMPLE, len(X_test))
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=sample_n, replace=False)
X_test_sample = X_test.iloc[sample_idx]

shap_values = explainer.shap_values(X_test_sample)
plt.figure()
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig(PLOTS / "shap_global.png", dpi=150); plt.close()

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

# ==== Persistência de artefatos ====
joblib.dump(model_for_proba, model_path)
joblib.dump(scaler, MODELS / "num_scaler.pkl")

metrics = {
    "roc_auc": float(roc),
    "average_precision": float(ap),
    "threshold": float(threshold),
    "fbeta": float(best["fbeta"]),
    "precision_at_threshold": float(best["precision"]),
    "recall_at_threshold": float(best["recall"]),
    "support_test": int(len(y_test)),
    "calibrated": bool(USE_CALIBRATION),
    "classification_report": report
}
with open(DOCS / "metrics_delay_model.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"💾 Artefatos salvos em {MODELS} e métricas em {DOCS/'metrics_delay_model.json'}")

# ==== Saída para Tableau: probabilidade por order_id + split ====
pred_test = test[["order_id"]].copy()
pred_test["prob_atraso"] = proba_test
pred_test["y_true"] = y_test.values
pred_test["split"] = "test"

proba_train = model_for_proba.predict_proba(X_train)[:, 1]
pred_train = train[["order_id"]].copy()
pred_train["prob_atraso"] = proba_train
pred_train["y_true"] = y_train.values
pred_train["split"] = "train"

pred_all = pd.concat([pred_train, pred_test], ignore_index=True)
pred_all.to_csv(PROCESSED / "predicted_delay.csv", index=False)
print(f"✅ Predições salvas em {PROCESSED/'predicted_delay.csv'}")

# ==== Tabela de decil (Lift) ====
q = pd.qcut(pred_test["prob_atraso"], 10, labels=False, duplicates="drop")  # 0 menor risco, 9 maior
decil = pd.DataFrame({"decile": q, "y": y_test})
lift = decil.groupby("decile")["y"].mean().rename("delay_rate").reset_index()
lift["share"] = decil.groupby("decile")["y"].size().values / len(decil)
lift.to_csv(DOCS / "decile_lift.csv", index=False)
print(f"📈 Decile lift salvo em {DOCS/'decile_lift.csv'}")

# ==== Métricas por segmento (diagnóstico) ====
def eval_segment(df_seg: pd.DataFrame, key: str):
    rows = []
    for val, d in df_seg.groupby(key):
        if d["y_true"].nunique() == 1:
            # AUC/AP não definidos quando só uma classe
            auc = np.nan
            apv = np.nan
        else:
            auc = roc_auc_score(d["y_true"], d["prob_atraso"])
            apv = average_precision_score(d["y_true"], d["prob_atraso"])
        rows.append({"segment": val, "n": len(d), "AUC": auc, "AP": apv})
    return pd.DataFrame(rows).sort_values(by=["AUC","AP"], ascending=[False, False])

pred_test_seg = test[["customer_state", "product_category_name"]].copy()
pred_test_seg["prob_atraso"] = proba_test
pred_test_seg["y_true"] = y_test.values

seg_state = eval_segment(pred_test_seg.rename(columns={"customer_state": "segment"}), "segment")
seg_cat = eval_segment(pred_test_seg.rename(columns={"product_category_name": "segment"}), "segment")

seg_state.to_csv(DOCS / "segment_metrics_customer_state.csv", index=False)
seg_cat.to_csv(DOCS / "segment_metrics_category.csv", index=False)
print("🗂️ Métricas por segmento salvas")

print("🎉 Treinamento v2 finalizado com sucesso!")