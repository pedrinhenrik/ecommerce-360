"""
Build features para previsão de atraso usando apenas tabelas do dataset Olist.
Saídas:
- data/processed/dataset_delay_features.parquet
- data/processed/predicted_delay_base.csv 
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---- Configuração de paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ---- Utilidades
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distância Haversine em km (vetorizado)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def safe_div(num, den, eps=1.0):
    """Divisão estável (evita zero)."""
    return num / (den + eps)

print("🚀 Iniciando build_features...")

# ---- Carregar dados
orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"
])
order_items = pd.read_csv(RAW / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
products = pd.read_csv(RAW / "olist_products_dataset.csv")
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
geoloc = pd.read_csv(RAW / "olist_geolocation_dataset.csv")

print("📥 Dados carregados com sucesso")

# ---- Coordenadas por CEP (prefixo)
geo_zip = (geoloc
           .groupby("geolocation_zip_code_prefix", as_index=False)
           .agg(lat=("geolocation_lat", "mean"),
                lng=("geolocation_lng", "mean")))

# ---- Enriquecimento sellers/clientes
sellers_geo = sellers.merge(geo_zip, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left") \
                     .rename(columns={"lat": "seller_lat", "lng": "seller_lng"}) \
                     .drop(columns=["geolocation_zip_code_prefix"])

customers_geo = customers.merge(geo_zip, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left") \
                         .rename(columns={"lat": "cust_lat", "lng": "cust_lng"}) \
                         .drop(columns=["geolocation_zip_code_prefix"])

print("🗺️ Coordenadas de sellers e clientes associadas")

# ---- Consolidado itens + produtos + sellers
items_full = (order_items
              .merge(products[[
                  "product_id", "product_category_name",
                  "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
              ]], on="product_id", how="left")
              .merge(sellers_geo[[
                  "seller_id", "seller_state", "seller_lat", "seller_lng"
              ]], on="seller_id", how="left"))

# ---- Consolidado com pedidos + clientes
items_orders = (items_full
                .merge(orders[[
                    "order_id", "customer_id",
                    "order_purchase_timestamp", "order_delivered_customer_date",
                    "order_estimated_delivery_date", "order_status"
                ]], on="order_id", how="left")
                .merge(customers_geo[[
                    "customer_id", "customer_state", "cust_lat", "cust_lng"
                ]], on="customer_id", how="left"))

# ---- Apenas entregues
items_orders = items_orders[items_orders["order_status"].eq("delivered")].copy()

print(f"📦 Base consolidada: {len(items_orders):,} registros entregues")

# ---- Features temporais
dtp = items_orders["order_purchase_timestamp"]
dte = items_orders["order_estimated_delivery_date"]
dtr = items_orders["order_delivered_customer_date"]

items_orders["dias_previstos_entrega"] = (dte - dtp).dt.days.clip(lower=0)
items_orders["dias_reais_entrega"] = (dtr - dtp).dt.days.clip(lower=0)
items_orders["atraso_dias"] = (items_orders["dias_reais_entrega"] - items_orders["dias_previstos_entrega"]).clip(-30, 60)
items_orders["y_atraso"] = (items_orders["atraso_dias"] > 0).astype(int)

# novo: mês numérico da compra
items_orders["mes_num"] = dtp.dt.month.astype("int8")

# ---- Features logísticas
items_orders["dist_km"] = haversine_km(items_orders["seller_lat"], items_orders["seller_lng"],
                                       items_orders["cust_lat"], items_orders["cust_lng"])

# ---- Produto
l = items_orders["product_length_cm"].clip(lower=0)
h = items_orders["product_height_cm"].clip(lower=0)
w = items_orders["product_width_cm"].clip(lower=0)
peso_kg = items_orders["product_weight_g"].clip(lower=0) / 1000.0
vol_m3 = (l * h * w) / 1e6
items_orders["peso_kg"] = peso_kg.astype(float)
items_orders["vol_m3"] = vol_m3.replace(0, np.nan)
items_orders["densidade"] = safe_div(items_orders["peso_kg"], items_orders["vol_m3"])

# ---- Econômicas
items_orders["frete_ratio"] = safe_div(items_orders["freight_value"], items_orders["price"])
items_orders["ticket"] = items_orders["price"].astype(float)

print("🛠️ Features criadas")

# ---- Dataset final
feature_cols = ["dias_previstos_entrega","dist_km","peso_kg","vol_m3","densidade","frete_ratio","ticket","mes_num"]
cat_cols = ["seller_state","customer_state","product_category_name"]
id_cols = ["order_id","order_item_id","seller_id","product_id"]

dataset = items_orders[id_cols + feature_cols + cat_cols + ["y_atraso","order_purchase_timestamp","atraso_dias"]].copy()

# ---- Tratamento de faltas
for c in feature_cols:
    dataset[c] = dataset[c].fillna(dataset[c].median())
for c in cat_cols:
    dataset[c] = dataset[c].fillna("missing").astype(str)

# ---- Salvar parquet com fallback CSV
out_parquet = PROCESSED / "dataset_delay_features.parquet"
out_csv_base = PROCESSED / "predicted_delay_base.csv"

try:
    dataset.to_parquet(out_parquet, index=False)
    print(f"✅ Features salvas em {out_parquet}")
except Exception as e:
    alt = PROCESSED / "dataset_delay_features.csv"
    dataset.to_csv(alt, index=False)
    print(f"⚠️ Parquet indisponível ({e.__class__.__name__}). CSV salvo em {alt}")

# Base mínima
base_cols = ["order_id","order_item_id","order_purchase_timestamp","atraso_dias","y_atraso"] + feature_cols + cat_cols
dataset[base_cols].to_csv(out_csv_base, index=False)
print(f"✅ Base mínima salva em {out_csv_base}")

print("🎉 build_features finalizado com sucesso!")