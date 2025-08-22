# scripts/save_predicted_delay.py
from pathlib import Path
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.types import BigInteger, Float, Text, DateTime

# ============== CONFIG BÁSICA ==============
# paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

CSV_DELAY = DATA / "predicted_delay.csv"
CSV_BASE  = DATA / "predicted_delay_base.csv"

# conexão (psycopg v3)
url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="Postgre2025!",
    host="localhost",
    port=5432,
    database="ecommerce_olist",
)
engine = create_engine(url, pool_pre_ping=True)


# ============== HELPERS ROBUSTOS ==============
def read_csv_robusto(path: Path) -> pd.DataFrame:
    """
    Lê CSV testando separadores comuns com o parser C (rápido e estável).
    Evita engine='python' para não conflitar com low_memory e para ganhar performance.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    tentativas = [
        {"sep": ",", "engine": "c"},
        {"sep": ";", "engine": "c"},
        {"sep": "\t", "engine": "c"},
    ]
    ultimo_erro = None
    for opt in tentativas:
        try:
            df = pd.read_csv(path, **opt)
            # Heurística: 1 coluna gigante sugere separador errado
            if df.shape[1] == 1 and ("," in df.columns[0] or ";" in df.columns[0]):
                raise ValueError("Possível separador incorreto (apenas 1 coluna detectada).")
            return df
        except Exception as e:
            ultimo_erro = e
    raise ValueError(f"Falha ao ler {path.name} com separadores comuns. Último erro: {ultimo_erro}")


def sanitize_dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes (sem acento, minúsculo, underscores), limita tamanho e resolve duplicatas.
    """
    import unicodedata, re

    def norm(c):
        c = str(c)
        c = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii")
        c = re.sub(r"[^\w]+", "_", c).strip("_").lower()
        return c[:60] if len(c) > 60 else c

    new_cols = []
    seen = {}
    for c in df.columns:
        base = norm(c) or "col"
        name = base
        k = 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen[name] = 1
        new_cols.append(name)
    df.columns = new_cols
    return df


def coerce_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte listas/dicts em JSON string; tenta numérico/datetime nas 'object'.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(50)
            # estruturas viram JSON
            if sample.apply(lambda x: isinstance(x, (dict, list, tuple, set))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (dict, list, tuple, set))
                    else x
                )
            else:
                # tenta numérico
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass
                # tenta datetime (usa se maioria for válida)
                try:
                    if df[col].dtype == "object":
                        dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                        if dt.notna().mean() > 0.9:
                            df[col] = dt
                except Exception:
                    pass
    return df


def build_dtype_map(df: pd.DataFrame):
    """
    Define dtype explícito para o Postgres (evita inferências ruins).
    """
    dtype = {}
    for col, d in df.dtypes.items():
        if pd.api.types.is_integer_dtype(d):
            dtype[col] = BigInteger()
        elif pd.api.types.is_float_dtype(d):
            dtype[col] = Float()
        elif pd.api.types.is_datetime64_any_dtype(d):
            dtype[col] = DateTime()
        else:
            dtype[col] = Text()
    return dtype


def print_overview(nome: str, df: pd.DataFrame, head_n: int = 3):
    print(f"\n[INFO] {nome} -> shape: {df.shape}")
    print(df.head(head_n))
    print(df.dtypes)


# ============== PIPELINE DE CARGA ==============
if __name__ == "__main__":
    # garantir schema
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS dwh"))

    # ---------- predicted_delay ----------
    df_delay = read_csv_robusto(CSV_DELAY)
    print_overview("predicted_delay.csv", df_delay)

    # normaliza tipos-chave (strings)
    for k in ["order_id", "order_item_id"]:
        if k in df_delay.columns:
            df_delay[k] = df_delay[k].astype(str)

    if df_delay.empty:
        raise ValueError("df_delay está vazio. Verifique predicted_delay.csv")

    df_delay.to_sql(
        "predicted_delay",
        engine, schema="dwh",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=10_000,
    )

    # ---------- predicted_delay_base (NARROW + JSON do resto) ----------
    df_base = read_csv_robusto(CSV_BASE)
    print_overview("predicted_delay_base.csv (bruto)", df_base)

    # higieniza nomes e deduplica
    df_base = sanitize_dedup_columns(df_base)
    # escalares e coerções
    df_base = coerce_scalars(df_base)

    # chaves como string
    for k in ["order_id", "order_item_id"]:
        if k in df_base.columns:
            df_base[k] = df_base[k].astype(str)

    if df_base.empty:
        raise ValueError("df_base está vazio após limpeza. Verifique predicted_delay_base.csv")

    # ---- COLUNAS ESSENCIAIS PARA O DASH ----
    # Ajuste os nomes abaixo conforme ficaram após sanitize_dedup_columns (snake_case).
    preferidas = [
        "order_id",
        "order_item_id",
        "y_true",                   # atraso real 0/1
        "prob_atraso",              # probabilidade prevista do modelo
        "price",                    # preço do item (se existir)
        "freight_value",            # frete (se existir)
        "shipping_limit_date",      # data limite
        "seller_state",             # UF vendedor
        "customer_state",           # UF cliente
        "product_category_name",    # categoria
        "dist_km",                  # distância (se existir)
        "peso_kg",                  # peso (se existir)
        "ticket",                   # ticket médio/valor (se existir)
    ]

    # mantém apenas as que existem
    essenciais = [c for c in preferidas if c in df_base.columns]
    resto_cols = [c for c in df_base.columns if c not in essenciais]

    # coluna JSON com o restante
    def row_to_json(row):
        d = {k: row[k] for k in resto_cols}
        # normaliza datetimes para ISO
        for k, v in d.items():
            if hasattr(v, "isoformat"):
                try:
                    d[k] = v.isoformat()
                except Exception:
                    d[k] = str(v)
        return json.dumps(d, ensure_ascii=False)

    if resto_cols:
        df_base["features_json"] = df_base.apply(row_to_json, axis=1)
        essenciais_out = essenciais + ["features_json"]
    else:
        essenciais_out = essenciais

    df_base_narrow = df_base[essenciais_out].copy()

    print_overview("predicted_delay_base (narrow)", df_base_narrow)

    dtype_map = build_dtype_map(df_base_narrow)
    df_base_narrow.to_sql(
        "predicted_delay_base",
        engine, schema="dwh",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=10_000,
        dtype=dtype_map,
    )

    # ---------- validação pós-carga ----------
    with engine.begin() as conn:
        cnt_delay = conn.execute(text("SELECT COUNT(*) FROM dwh.predicted_delay")).scalar_one()
        cnt_base  = conn.execute(text("SELECT COUNT(*) FROM dwh.predicted_delay_base")).scalar_one()

    print(f"\n✅ Linhas em dwh.predicted_delay: {cnt_delay}")
    print(f"✅ Linhas em dwh.predicted_delay_base: {cnt_base}\n")

    print("✅ dwh.predicted_delay e dwh.predicted_delay_base carregadas com sucesso!")
