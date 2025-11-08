import pandas as pd
from data_loader import load_data
from preprocessing import create_target_variable, clean_data

def consolidate_match_data(df_full: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida df_full e df_stats, colocando estatísticas do mandante e
    visitante na mesma linha.
    """
    print("ℹ️ Consolidando dados de estatísticas (Issue #6 - Parte 1)...")
    
    # 1. Separar estatísticas de mandante e visitante
    # Em df_stats, 'clube' é o nome do time. Comparar com 'mandante' e 'visitante' de df_full
    df_full_merged = df_full.merge(
        df_stats.add_prefix('mandante_'), 
        left_on=['id', 'mandante'], 
        right_on=['mandante_partida_id', 'mandante_clube'],
        how='left'
    )

    df_full_merged = df_full_merged.merge(
        df_stats.add_prefix('visitante_'),
        left_on=['id', 'visitante'],
        right_on=['visitante_partida_id', 'visitante_clube'],
        how='left'
    )
    
    # 2. Calcular pontos (essencial para médias móveis)
    def assign_points(row):
        if row['resultado'] == 'Vitoria_Mandante':
            return 3, 0
        if row['resultado'] == 'Vitoria_Visitante':
            return 0, 3
        if row['resultado'] == 'Empate':
            return 1, 1
        return None, None

    df_full_merged[['pontos_mandante', 'pontos_visitante']] = df_full_merged.apply(
        assign_points, axis=1, result_type='expand'
    )
    
    print("✅ Dados consolidados. Estatísticas e pontos por partida criados.")
    return df_full_merged


def create_rolling_features(df_partidas: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de média móvel (rolling) para a forma dos times.
    !! Esta é a etapa crucial para evitar Data Leakage !!
    """
    print("ℹ️ Criando features de média móvel (Issue #6 - Parte 2)...")
    
    # Garantir que as partidas estejam em ordem cronológica
    df_partidas = df_partidas.sort_values(by='data')

    # Lista de estatísticas que queremos acompanhar
    stats_cols = [
        'pontos',
        'mandante_placar', 'visitante_placar', # Serão 'gols_feitos' e 'gols_sofridos'
        'mandante_chutes', 'mandante_chutes_no_alvo', 'mandante_posse_de_bola',
        'mandante_faltas', 'mandante_cartao_amarelo', 'mandante_escanteios'
    ]
    # Renomear colunas de stats para algo genérico (ex: 'gols_feitos')
    # Isso é complexo. Vamos simplificar e usar apenas as colunas já existentes.

    # 1. Preparar dados longos para cálculo de rolling
    # Criar um DataFrame para mandantes e outro para visitantes
    df_mandante = df_partidas.copy()
    df_visitante = df_partidas.copy()

    # Renomear colunas para genérico (ex: 'clube', 'gols_feitos', 'gols_sofridos')
    cols_rename_mandante = {
        'mandante': 'clube', 'visitante': 'adversario',
        'pontos_mandante': 'pontos',
        'mandante_placar': 'gols_feitos', 'visitante_placar': 'gols_sofridos',
        'mandante_chutes': 'chutes', 'mandante_chutes_no_alvo': 'chutes_no_alvo',
        'mandante_posse_de_bola': 'posse_de_bola', 'mandante_faltas': 'faltas'
    }
    
    cols_rename_visitante = {
        'visitante': 'clube', 'mandante': 'adversario',
        'pontos_visitante': 'pontos',
        'visitante_placar': 'gols_feitos', 'mandante_placar': 'gols_sofridos',
        'visitante_chutes': 'chutes', 'visitante_chutes_no_alvo': 'chutes_no_alvo',
        'visitante_posse_de_bola': 'posse_de_bola', 'visitante_faltas': 'faltas'
    }

    df_mandante = df_mandante.rename(columns=cols_rename_mandante)
    df_visitante = df_visitante.rename(columns=cols_rename_visitante)
    
    # Manter apenas colunas relevantes
    keep_cols = ['id', 'data', 'clube', 'adversario', 'pontos', 'gols_feitos', 'gols_sofridos',
                   'chutes', 'chutes_no_alvo', 'posse_de_bola', 'faltas']
    
    df_mandante = df_mandante[keep_cols]
    df_visitante = df_visitante[keep_cols]

    # Combinar tudo em um dataframe 'longo'
    df_long = pd.concat([df_mandante, df_visitante]).sort_values(by='data')

    # 2. Calcular médias móveis
    # Usamos shift(1) para garantir que usamos dados APENAS ANTERIORES à partida (EVITA LEAKAGE)
    # A primeira partida de um time na janela terá NaN (o que é correto)
    N_GAMES = 5 # Número de jogos para a média móvel
    
    # Agrupar por clube
    grouped = df_long.groupby('clube')
    
    features = ['pontos', 'gols_feitos', 'gols_sofridos', 'chutes', 'chutes_no_alvo', 'posse_de_bola', 'faltas']
    
    for col in features:
        # shift(1) pega o dado da partida anterior
        # rolling(N_GAMES, min_periods=1) calcula a média dos últimos N jogos
        # min_periods=1 garante que times com < N jogos ainda tenham uma média
        df_long[f'media_{col}_ult_{N_GAMES}'] = grouped[col].shift(1).rolling(N_GAMES, min_periods=1).mean()

    print(f"✅ Features de média móvel (últimos {N_GAMES} jogos) calculadas.")
    
    # 3. Juntar features de volta no dataframe original
    # Separar as features do mandante e do visitante
    df_features_mandante = df_long[df_long['id'].isin(df_partidas['id'])].add_prefix('mandante_')
    df_features_visitante = df_long[df_long['id'].isin(df_partidas['id'])].add_prefix('visitante_')

    # Juntar features do mandante
    df_final = df_partidas.merge(
        df_features_mandante.drop_duplicates(subset=['mandante_id', 'mandante_clube']),
        left_on=['id', 'mandante'],
        right_on=['mandante_id', 'mandante_clube'],
        how='left'
    )
    
    # Juntar features do visitante
    df_final = df_final.merge(
        df_features_visitante.drop_duplicates(subset=['visitante_id', 'visitante_clube']),
        left_on=['id', 'visitante'],
        right_on=['visitante_id', 'visitante_clube'],
        how='left'
    )
    
    print("✅ Features de rolling unidas ao dataframe final.")
    
    # 4. Limpeza Final
    # Remover colunas que causam Data Leakage (as estatísticas DA PRÓPRIA PARTIDA)
    cols_to_drop_leakage = [
        'mandante_placar', 'visitante_placar', 'pontos_mandante', 'pontos_visitante',
        'mandante_chutes', 'mandante_chutes_no_alvo', 'mandante_posse_de_bola', 'mandante_passes',
        'mandante_precisao_passes', 'mandante_faltas', 'mandante_cartao_amarelo',
        'mandante_cartao_vermelho', 'mandante_impedimentos', 'mandante_escanteios',
        'visitante_chutes', 'visitante_chutes_no_alvo', 'visitante_posse_de_bola', 'visitante_passes',
        'visitante_precisao_passes', 'visitante_faltas', 'visitante_cartao_amarelo',
        'visitante_cartao_vermelho', 'visitante_impedimentos', 'visitante_escanteios',
    ]
    
    # Remover colunas duplicadas ou de referência
    cols_to_drop_reference = [
        'vencedor', 'data', # Data já foi usada para ordenar
        'mandante_partida_id', 'mandante_clube', 'mandante_rodata',
        'visitante_partida_id', 'visitante_clube', 'visitante_rodata',
    ]

    # Precisamos ter cuidado para não dropar colunas que não existem
    # (ex: 'mandante_passes' pode não existir se não estava em df_stats)
    cols_to_drop = [col for col in (cols_to_drop_leakage + cols_to_drop_reference) if col in df_final.columns]
    
    df_final = df_final.drop(columns=cols_to_drop)

    # Lidar com NaNs (partidas iniciais onde a média móvel não existe)
    # Preencher com 0 é uma estratégia segura
    df_final = df_final.fillna(0)
    
    print("✅ Limpeza de colunas com data leakage concluída.")
    return df_final


# --- Bloco de Teste ---
if __name__ == "__main__":
    print("--- Testando o módulo feature_engineering.py ---")
    
    # 1. Carga e Limpeza (Issues #3, #4, #5)
    df_full, df_stats, df_gols, df_cartoes = load_data()
    df_full = create_target_variable(df_full)
    df_full_clean, df_stats_clean = clean_data(df_full, df_stats)

    # 2. Consolidação (Issue #6 - Parte 1)
    df_partidas = consolidate_match_data(df_full_clean, df_stats_clean)
    print("\n--- Verificação Pós-Consolidação ---")
    print(df_partidas[['id', 'mandante', 'visitante', 'mandante_chutes', 'visitante_chutes', 'pontos_mandante']].head(3))

    # 3. Engenharia de Features (Issue #6 - Parte 2)
    df_modelo = create_rolling_features(df_partidas)
    
    print("\n--- Verificação Final (Pronto para o Modelo) ---")
    pd.set_option('display.max_columns', None) # Mostrar todas as colunas
    print(df_modelo.head(5))
    
    print("\n--- Colunas Finais ---")
    print(list(df_modelo.columns))
    
    print("\n--- Verificação de Nulos no DataFrame Final ---")
    print(f"Existem nulos? {df_modelo.isnull().values.any()}")