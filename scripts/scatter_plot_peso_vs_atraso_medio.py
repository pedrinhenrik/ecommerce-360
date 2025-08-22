# scripts/build_peso_atraso_and_plot.py
import os
from pathlib import Path
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ------------------ Paths & CLI ------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_FIGS = ROOT / "data" / "Images"

DEFAULT_ORDERS   = DATA_RAW / "olist_orders_dataset.csv"
DEFAULT_ITEMS    = DATA_RAW / "olist_order_items_dataset.csv"
DEFAULT_PRODUCTS = DATA_RAW / "olist_products_dataset.csv"
DEFAULT_OUTCSV   = DATA_PROC / "peso_atraso_base.csv"

parser = argparse.ArgumentParser(description="Gera base peso x atraso e cria os gráficos (493x200 e 986x200).")
parser.add_argument("--orders",   default=str(DEFAULT_ORDERS),   help="CSV olist_orders_dataset.csv")
parser.add_argument("--items",    default=str(DEFAULT_ITEMS),    help="CSV olist_order_items_dataset.csv")
parser.add_argument("--products", default=str(DEFAULT_PRODUCTS), help="CSV olist_products_dataset.csv")
parser.add_argument("--outcsv",   default=str(DEFAULT_OUTCSV),   help="CSV de saída processado")
parser.add_argument("--outdir",   default=str(OUT_FIGS),         help="Pasta para as imagens PNG")
parser.add_argument("--bins",     type=int, default=20,          help="Bins para média por peso")
parser.add_argument("--min-count",type=int, default=10,          help="Mínimo de pontos por bin")
args = parser.parse_args()

# ------------------ Load ------------------
orders   = pd.read_csv(args.orders, parse_dates=["order_delivered_customer_date",
                                                 "order_estimated_delivery_date"], low_memory=False)
items    = pd.read_csv(args.items, low_memory=False)
products = pd.read_csv(args.products, low_memory=False)

# ------------------ Prep: atraso por pedido ------------------
# atraso_dias = (entrega_real - entrega_prevista) em dias
orders["atraso_dias"] = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.days
# y_atraso = 1 se atrasou (dias > 0), 0 caso contrário (<=0 ou NA)
orders["y_atraso"] = (orders["atraso_dias"] > 0).astype(int)

# Mantém somente pedidos com data de entrega e previsão válidas
orders_valid = orders.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"]).copy()
orders_valid = orders_valid[["order_id", "atraso_dias", "y_atraso"]]

# ------------------ Prep: peso por pedido ------------------
# peso do produto em kg (algumas bases têm em gramas)
if "product_weight_kg" in products.columns:
    products["peso_kg"] = products["product_weight_kg"]
elif "product_weight_g" in products.columns:
    products["peso_kg"] = products["product_weight_g"] / 1000.0
else:
    raise ValueError("Coluna de peso não encontrada em products (esperado product_weight_g ou product_weight_kg).")

# junta items->products para pegar peso por item
items_weight = items.merge(
    products[["product_id", "peso_kg"]],
    on="product_id",
    how="left",
    validate="m:1",
)

# alguns produtos podem ter peso ausente; remove/zera
items_weight["peso_kg"] = pd.to_numeric(items_weight["peso_kg"], errors="coerce").fillna(0.0)

# peso total por pedido = soma do peso dos itens daquele pedido (order_item_id é 1 por linha)
peso_por_pedido = (
    items_weight.groupby("order_id", as_index=False)["peso_kg"]
    .sum()
    .rename(columns={"peso_kg": "peso_total_kg"})
)

# ------------------ Join final ------------------
base = orders_valid.merge(peso_por_pedido, on="order_id", how="inner", validate="1:1")

# Limpeza: remove linhas com peso negativo/NaN e atraso NaN
base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=["peso_total_kg", "atraso_dias"])

# ------------------ Salvar CSV ------------------
Path(args.outcsv).parent.mkdir(parents=True, exist_ok=True)
base.to_csv(args.outcsv, index=False)
print(f"✅ CSV gerado: {args.outcsv}  (linhas: {len(base)})")

# ------------------ Plot Helper ------------------
def make_scatter(df_in: pd.DataFrame, width_px: int, height_px: int, outfile: Path,
                 bins: int = 20, min_count: int = 10):
    # Fonte Segoe UI Bold
    try:
        matplotlib.rcParams["font.family"] = "Segoe UI"
        matplotlib.rcParams["font.weight"] = "bold"
    except Exception as e:
        warnings.warn(f"Não foi possível aplicar 'Segoe UI'. Usando padrão. ({e})")

    df = df_in.copy()
    # Considera somente pedidos com atraso para o cálculo de atraso médio
    df = df[df["y_atraso"] == 1].copy()
    df = df[df["atraso_dias"] > 0]

    # Limita outliers no 99º percentil para o eixo X
    p99 = df["peso_total_kg"].quantile(0.99)
    df["peso_total_kg_cap"] = np.clip(df["peso_total_kg"], a_min=df["peso_total_kg"].min(), a_max=p99)

    # Amostra para o scatter (evita pesar)
    max_pts = 5000 if width_px >= 900 else 2500
    sample = df.sample(min(len(df), max_pts), random_state=42)

    # Bins por peso (linspace para bins uniformes)
    edges = np.linspace(df["peso_total_kg_cap"].min(), df["peso_total_kg_cap"].max(), bins + 1)
    idx = np.digitize(df["peso_total_kg_cap"], edges) - 1
    mean_x, mean_y = [], []
    for i in range(bins):
        sel = idx == i
        if sel.sum() >= min_count:
            mean_x.append(df.loc[sel, "peso_total_kg_cap"].mean())
            mean_y.append(df.loc[sel, "atraso_dias"].mean())

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Scatter (sem definir cor explicitamente)
    ax.scatter(sample["peso_total_kg_cap"], sample["atraso_dias"],s=10, alpha=0.35, color="#2e6bd6")

    # Linha de média por bin
    if mean_x: ax.plot(mean_x, mean_y, linewidth=2.2, color="#2e6bd6")

    ax.set_title("Peso dos pedidos vs Atraso médio (somente atrasados)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Peso total do pedido (kg)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Atraso médio (dias)", fontsize=9, fontweight="bold")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Gráfico salvo: {outfile}")

# ------------------ Gerar as imagens ------------------
OUT_FIGS.mkdir(parents=True, exist_ok=True)
df_plot = base[["peso_total_kg", "atraso_dias", "y_atraso"]].copy()

make_scatter(df_plot, 493, 200, Path(args.outdir) / "peso_vs_atraso_493x200.png",
             bins=args.bins, min_count=args.min_count)
make_scatter(df_plot, 986, 200, Path(args.outdir) / "peso_vs_atraso_986x200.png",
             bins=args.bins, min_count=args.min_count)

print("✅ Processo concluído.")
