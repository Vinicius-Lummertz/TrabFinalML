import pandas as pd
from typing import Tuple

# ---
# ETAPA 5: Definição do Alvo
# ---
def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna alvo 'Resultado' com base na coluna 'Vencedor'.
    
    Categorias:
    - Vitoria_Mandante
    - Vitoria_Visitante
    - Empate
    """
    print("ℹ️ Criando variável alvo 'Resultado'...")

    def map_vencedor(row):
        if row['vencedor'] == '-':
            return 'Empate'
        if row['vencedor'] == row['mandante']:
            return 'Vitoria_Mandante'
        if row['vencedor'] == row['visitante']:
            return 'Vitoria_Visitante'
        return None # Caso de segurança, não deve ocorrer

    # Renomeando colunas para minúsculas para consistência
    df.columns = df.columns.str.lower()
    
    df['resultado'] = df.apply(map_vencedor, axis=1)
    
    # Verifica se a criação foi bem-sucedida
    if df['resultado'].isnull().any():
        print("⚠️ Alerta: Existem nulos na coluna 'Resultado'. Verifique a lógica.")
        
    print("✅ Variável alvo 'Resultado' criada.")
    return df

# ---
# ETAPA 4: Limpeza e Tratamento
# ---
def clean_data(df_full: pd.DataFrame, df_stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Limpa os DataFrames full e stats com base nas decisões do diagnóstico.
    """
    print("ℹ️ Iniciando limpeza (Issue #4)...")
    
    # 1. (Full) Dropar colunas com mais de 50% de nulos
    cols_to_drop = ['formacao_mandante', 'formacao_visitante', 'tecnico_mandante', 'tecnico_visitante']
    df_full_clean = df_full.drop(columns=cols_to_drop)
    print(f"✅ Colunas dropadas: {cols_to_drop}")

    # 2. (Full) Converter 'data' para datetime
    df_full_clean['data'] = pd.to_datetime(df_full_clean['data'], format='%d/%m/%Y')
    
    # 3. (Full) Engenharia de feature simples: 'Turno' a partir da 'hora'
    # Extrai a hora (ex: "16:00" -> 16)
    df_full_clean['hora_numerica'] = pd.to_numeric(df_full_clean['hora'].str.split(':').str[0], errors='coerce')
    
    def map_turno(hora):
        if pd.isnull(hora):
            return 'Noite' # Assume 'Noite' como padrão se a hora for nula
        if hora <= 12:
            return 'Manha'
        if hora <= 18:
            return 'Tarde'
        return 'Noite'

    df_full_clean['turno'] = df_full_clean['hora_numerica'].apply(map_turno)
    
    # Remove colunas originais que não são mais necessárias
    df_full_clean = df_full_clean.drop(columns=['hora', 'hora_numerica'])
    print("✅ Coluna 'data' convertida para datetime e 'turno' criada.")

    # 4. (Stats) Renomear colunas para minúsculas
    df_stats.columns = df_stats.columns.str.lower()

    # 5. (Stats) Preencher NaNs com 0 (assumindo que "não coletado" = 0)
    stats_cols_to_fill = ['posse_de_bola', 'precisao_passes', 'chutes', 'chutes_no_alvo']
    
    # Criando cópia para evitar SettingWithCopyWarning
    df_stats_clean = df_stats.copy()
    
    for col in stats_cols_to_fill:
        if col in df_stats_clean.columns:
            df_stats_clean[col] = df_stats_clean[col].fillna(0)
            
    print(f"✅ NaNs preenchidos com 0 em 'df_stats' (ex: posse_de_bola).")
    
    return df_full_clean, df_stats_clean


# --- Bloco de Teste ---
if __name__ == "__main__":
    from data_loader import load_data
    
    print("--- Testando o módulo preprocessing.py ---")
    df_full, df_stats, df_gols, df_cartoes = load_data()
    
    if df_full is not None:
        # 1. Teste da Issue #5
        df_full = create_target_variable(df_full)
        print("\n--- Verificação da Variável Alvo ---")
        print(df_full[['vencedor', 'mandante', 'resultado']].head(3))
        
        # Mostra a distribuição das classes (importante para o modelo)
        print("\nDistribuição das Classes:")
        print(df_full['resultado'].value_counts(normalize=True) * 100)
        
        # 2. Teste da Issue #4
        df_full_clean, df_stats_clean = clean_data(df_full, df_stats)
        
        print("\n--- Verificação Pós-Limpeza (Full) ---")
        df_full_clean.info() # Deve mostrar 0 nulos (exceto 'resultado' se houver falha)
        
        print("\n--- Verificação Pós-Limpeza (Stats) ---")
        # Deve mostrar '0' onde antes era 'NaN' (ex: posse_de_bola linha 0)
        print(df_stats_clean[df_stats_clean['partida_id'] == 1].head())