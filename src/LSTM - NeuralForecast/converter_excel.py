import pandas as pd
from pathlib import Path

PASTA_ENTRADA = Path("data/Usina/Dados_coletados")
PASTA_SAIDA = Path("data/Usina/Dados_coletados_csv")
ARQUIVO_CONSOLIDADO = PASTA_SAIDA / "dados_consolidados_2024_2025.csv"

COLUNAS_PADRAO = [
    "Date",
    "Plant",
    "Classification",
    "Capacity(kW)",
    "Generation(kWh)",
    "Income",
]


def encontrar_cabecalho(df_raw: pd.DataFrame) -> int | None:
    for i, row in df_raw.iterrows():
        valores = [str(v).strip().lower() for v in row.tolist()]
        if "date" in valores and "generation(kwh)" in valores:
            return i
    return None


def limpar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    df.columns = [str(c).strip() for c in df.columns]

    renomear = {
        "Income()": "Income",
        "Income": "Income",
        "Capacity (kW)": "Capacity(kW)",
        "Generation (kWh)": "Generation(kWh)",
    }
    df = df.rename(columns=renomear)

    cols_existentes = [c for c in COLUNAS_PADRAO if c in df.columns]
    df = df[cols_existentes]

    if "Plant" in df.columns:
        df = df[
            ~df["Plant"].astype(str).str.strip().str.upper().eq("TOTAL")
        ]

    if "Date" in df.columns:
        df = df[
            ~df["Date"].astype(str).str.strip().str.upper().eq("TOTAL")
        ]

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    colunas_numericas = ["Capacity(kW)", "Generation(kWh)", "Income"]
    for col in colunas_numericas:
        if col in df.columns:
            serie = df[col].astype(str).str.strip()
            serie = serie.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)
            serie = serie.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(serie, errors="coerce")

    if "Date" in df.columns:
        df = df[df["Date"].notna()]

    df = df.reset_index(drop=True)
    return df


def processar_arquivo(arquivo: Path) -> pd.DataFrame:
    df_raw = pd.read_excel(arquivo, header=None)

    idx_header = encontrar_cabecalho(df_raw)
    if idx_header is None:
        raise ValueError("Não foi possível localizar a linha de cabeçalho.")

    cabecalho = df_raw.iloc[idx_header].tolist()
    df = df_raw.iloc[idx_header + 1:].copy()
    df.columns = cabecalho

    df = limpar_dataframe(df)
    return df


def main() -> None:
    arquivos = list(PASTA_ENTRADA.rglob("*.xls")) + list(PASTA_ENTRADA.rglob("*.xlsx"))

    if not arquivos:
        print("Nenhum arquivo Excel encontrado.")
        return

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    dfs_consolidados = []

    for arquivo in arquivos:
        try:
            print(f"Processando: {arquivo}")
            df_limpo = processar_arquivo(arquivo)

            rel_path = arquivo.relative_to(PASTA_ENTRADA).with_suffix(".csv")
            destino = PASTA_SAIDA / rel_path
            destino.parent.mkdir(parents=True, exist_ok=True)

            df_limpo.to_csv(destino, index=False, encoding="utf-8-sig")
            dfs_consolidados.append(df_limpo)

            print(f"Salvo CSV individual em: {destino}")

        except Exception as e:
            print(f"Erro em {arquivo.name}: {e}")

    if not dfs_consolidados:
        print("Nenhum arquivo pôde ser processado.")
        return

    df_final = pd.concat(dfs_consolidados, ignore_index=True)

    colunas_remover = ["Plant", "Classification", "source_file", "source_year_folder"]
    df_final = df_final.drop(columns=[c for c in colunas_remover if c in df_final.columns])

    if "Date" in df_final.columns:
        df_final = df_final.sort_values("Date")

    df_final = df_final.reset_index(drop=True)
    df_final.to_csv(ARQUIVO_CONSOLIDADO, index=False, encoding="utf-8-sig")

    print(f"✅ Arquivo consolidado salvo em: {ARQUIVO_CONSOLIDADO}")
    print(f"Total de linhas: {len(df_final)}")


if __name__ == "__main__":
    main()