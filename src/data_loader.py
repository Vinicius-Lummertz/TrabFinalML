import pandas as pd
from pathlib import Path
import sys
from typing import Tuple, Optional

# Encontra o caminho raiz do projeto (a pasta que contém 'src' e 'data')
# Path(__file__) é o caminho deste script
# .resolve() torna o caminho absoluto
# .parent é a pasta 'src'
# .parent é a raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# Nomes dos arquivos esperados
FILE_NAMES = {
    'full': 'campeonato-brasileiro-full.csv',
    'stats': 'campeonato-brasileiro-estatisticas-full.csv',
    'gols': 'campeonato-brasileiro-gols.csv',
    'cartoes': 'campeonato-brasileiro-cartoes.csv'
}

def load_data() -> Tuple[Optional[pd.DataFrame], ...]:
    """
    Carrega os 4 datasets do Brasileirão a partir da pasta /data.

    Assume que os arquivos CSV baixados do Kaggle estão localizados 
    na pasta /data/ na raiz do projeto.

    Retorna:
        tuple: Uma tupla contendo os 4 DataFrames (full, stats, gols, cartoes).
               Retorna (None, None, None, None) se qualquer arquivo não for encontrado.
    """
    dataframes = {}
    try:
        print(f"ℹ️ Tentando carregar dados da pasta: {DATA_PATH}")
        
        for key, name in FILE_NAMES.items():
            file_path = DATA_PATH / name
            
            if not file_path.exists():
                print(f"❌ ERRO: Arquivo não encontrado: {file_path}")
                print("Por favor, baixe o dataset do Kaggle e coloque os 4 CSVs na pasta /data.")
                # Retorna 4 Nones para desempacotamento seguro
                return None, None, None, None
            
            dataframes[key] = pd.read_csv(file_path)
            print(f"✅ Sucesso: Arquivo '{name}' carregado.")

        # Retorna na ordem definida para desempacotamento
        return (
            dataframes['full'],
            dataframes['stats'],
            dataframes['gols'],
            dataframes['cartoes']
        )

    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado ao carregar os dados: {e}")
        return None, None, None, None

# --- Bloco de Teste ---
# Este bloco só será executado se você rodar o script diretamente
# (ex: python src/data_loader.py)
if __name__ == "__main__":
    print("--- Testando o módulo data_loader.py ---")
    
    df_full, df_stats, df_gols, df_cartoes = load_data()
    
    if df_full is not None:
        print("\n--- Verificação dos DataFrames Carregados ---")
        
        print(f"\n[Full] Shape: {df_full.shape}")
        print(df_full.head(2))
        
        print(f"\n[Stats] Shape: {df_stats.shape}")
        print(df_stats.head(2))
        
        print(f"\n[Gols] Shape: {df_gols.shape}")
        print(df_gols.head(2))
        
        print(f"\n[Cartões] Shape: {df_cartoes.shape}")
        print(df_cartoes.head(2))
        
        print("\n--- Verificação de Nulos (Full) ---")
        # .info() é ótimo para uma visão rápida de nulos e tipos
        df_full.info()