from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# --- conexão (psycopg v3) ---
url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="Postgre2025!",
    host="localhost",
    port=5432,
    database="ecommerce_olist",
)
engine = create_engine(url, pool_pre_ping=True)

# --- garantir schema ---
with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS dwh"))

# --- ler dados de previsão ---
q = "SELECT order_id, prob_atraso, y_true, split FROM dwh.predicted_delay"
df = pd.read_sql(q, engine)

# sanity
df = df.dropna(subset=["prob_atraso", "y_true"]).copy()
df["y_true"] = df["y_true"].astype(int)

def safe_auc(y, p):
    # precisa ter as duas classes
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

def safe_ap(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))

rows = []
now = datetime.now(timezone.utc).isoformat()

# métricas por split
for s, d in df.groupby("split", dropna=False):
    rows.append({
        "metric": "roc_auc",
        "value": safe_auc(d["y_true"], d["prob_atraso"]),
        "split": str(s),
        "n": int(len(d)),
        "positives": int((d["y_true"]==1).sum()),
        "negatives": int((d["y_true"]==0).sum()),
        "computed_at_utc": now,
    })
    rows.append({
        "metric": "average_precision",
        "value": safe_ap(d["y_true"], d["prob_atraso"]),
        "split": str(s),
        "n": int(len(d)),
        "positives": int((d["y_true"]==1).sum()),
        "negatives": int((d["y_true"]==0).sum()),
        "computed_at_utc": now,
    })

# métricas overall
rows.append({
    "metric": "roc_auc",
    "value": safe_auc(df["y_true"], df["prob_atraso"]),
    "split": "overall",
    "n": int(len(df)),
    "positives": int((df["y_true"]==1).sum()),
    "negatives": int((df["y_true"]==0).sum()),
    "computed_at_utc": now,
})
rows.append({
    "metric": "average_precision",
    "value": safe_ap(df["y_true"], df["prob_atraso"]),
    "split": "overall",
    "n": int(len(df)),
    "positives": int((df["y_true"]==1).sum()),
    "negatives": int((df["y_true"]==0).sum()),
    "computed_at_utc": now,
})

metrics_df = pd.DataFrame(rows)

# --- gravar no Postgres ---
metrics_df.to_sql(
    "model_metrics",
    engine,
    schema="dwh",
    if_exists="replace",
    index=False,
    method="multi",
    chunksize=10_000,
)

print("✅ dwh.model_metrics atualizado")
print(metrics_df)