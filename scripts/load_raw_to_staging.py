import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

def ensure_schema_exists(schema_name):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

def load_csvs_to_staging(folder, schema):
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            try:
                df = pd.read_csv(path, encoding='latin1')  # <- aqui
            except UnicodeDecodeError:
                print(f"❌ Erro de codificação ao ler {file}.")
                continue
            table_name = "staging_" + file.replace("_dataset.csv", "").replace(".csv", "")
            print(f"✅ Carregando: {file} → {schema}.{table_name} ({len(df)} linhas)")
            df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False, chunksize=10000)
    print("🎉 Todos os arquivos foram carregados com sucesso!")

if __name__ == "__main__":
    ensure_schema_exists("staging_olist")
    load_csvs_to_staging("data/raw", "staging_olist")