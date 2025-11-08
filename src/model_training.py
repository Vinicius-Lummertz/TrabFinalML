import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# --- IMPORTAÇÃO ADICIONADA ---
from sklearn.utils.class_weight import compute_sample_weight # Para ponderar o XGBoost
import warnings
import numpy as np

# Ignorar warnings de convergência
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---
# 1. Carregar e Processar Todos os Dados
# ---
from data_loader import load_data
from preprocessing import create_target_variable, clean_data
from feature_engineering import consolidate_match_data, create_rolling_features

def get_final_data():
    """
    Executa todo o pipeline de ETL (Extract, Transform, Load)
    para obter os dados prontos para o modelo.
    """
    print("ℹ️ Executando pipeline completo de ETL...")
    df_full, df_stats, df_gols, df_cartoes = load_data()
    if df_full is None:
        print("❌ Falha no carregamento. Abortando.")
        return None
    
    df_full = create_target_variable(df_full)
    df_full_clean, df_stats_clean = clean_data(df_full, df_stats)
    
    df_partidas = consolidate_match_data(df_full_clean, df_stats_clean)
    df_modelo = create_rolling_features(df_partidas)
    
    print("✅ Pipeline de ETL concluído. Dados prontos para modelagem.")
    return df_modelo

# ---
# 2. Definir Pipeline de Pré-processamento (Issue #8)
# ---
def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Identifica colunas numéricas e categóricas e cria o ColumnTransformer.
    """
    print("ℹ️ Criando pipeline de pré-processamento (Issue #8)...")
    
    features = X.columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = [col for col in numeric_features if col not in ['id']]
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"  Numéricas ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"  Categóricas ({len(categorical_features)}): {list(categorical_features)}")

    # Criar os transformadores
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    def to_string(X):
        return X.astype(str)

    categorical_transformer = Pipeline(steps=[
        ('tostring', FunctionTransformer(to_string)), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
    ])

    # Criar o ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )
    
    print("✅ Preprocessor criado.")
    return preprocessor

# ---
# 3. Treinar Modelos (Issues #9 e #10 com Ponderação)
# ---
def train_baseline_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Cria e treina o modelo baseline (LogisticRegression) com ponderação de classe.
    """
    print("\nℹ️ Treinando modelo Baseline (Regressão Logística Ponderada) (Issue #9)...")
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced' # <-- CORREÇÃO: Lidar com desbalanceamento
        ))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    print("✅ Modelo Baseline Ponderado treinado.")
    return model_pipeline

def train_advanced_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Cria e treina o modelo avançado (XGBoost) com ponderação de classe.
    (Issue #10)
    """
    print("\nℹ️ Treinando modelo Avançado (XGBoost Ponderado) (Issue #10)...")
    
    # O XGBoost precisa que os alvos (y) sejam numéricos (0, 1, 2)
    target_map = {'Empate': 0, 'Vitoria_Mandante': 1, 'Vitoria_Visitante': 2}
    y_train_mapped = y_train.map(target_map)
    
    # --- CORREÇÃO: Calcular pesos para o XGBoost ---
    # Isso dá mais peso para 'Empate' e 'Vitoria_Visitante'
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train_mapped
    )
    print("  Pesos de classe calculados para o XGBoost.")
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='multi:softmax', 
            num_class=3,             
            random_state=42,
            n_estimators=100,        
            max_depth=5,             
            learning_rate=0.1
        ))
    ])
    
    # --- CORREÇÃO: Passar os pesos para o .fit() ---
    # O pipeline passa os pesos para o passo 'classifier'
    model_pipeline.fit(X_train, y_train_mapped, classifier__sample_weight=sample_weights)
    
    print("✅ Modelo Avançado Ponderado treinado.")
    return model_pipeline, target_map


# --- Bloco de Teste ---
if __name__ == "__main__":
    print("--- Testando o módulo model_training.py ---")
    
    # 1. Obter dados
    df_modelo = get_final_data()
    
    if df_modelo is not None:
        # 2. Separar X e y
        y = df_modelo['resultado']
        X = df_modelo.drop(columns=['resultado', 'id'], errors='ignore')
        
        # 3. Dividir em Treino e Teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y 
        )
        
        print(f"  Tamanho Treino: {X_train.shape[0]} amostras")
        print(f"  Tamanho Teste: {X_test.shape[0]} amostras")

        # 4. Criar Pré-processador (baseado APENAS nos dados de treino)
        preprocessor = create_preprocessor(X_train)
        
        # 5. Treinar e Avaliar Modelo Baseline (Issue #9)
        baseline_model = train_baseline_model(preprocessor, X_train, y_train)
        
        print("\n--- Avaliação do Modelo Baseline (Regressão Logística Ponderada) ---")
        y_pred_baseline = baseline_model.predict(X_test)
        accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
        print(f"\nAcurácia (Baseline Ponderado): {accuracy_baseline * 100:.2f} %")
        print("\nRelatório de Classificação (Baseline Ponderado):")
        print(classification_report(y_test, y_pred_baseline))

        # 6. Treinar e Avaliar Modelo Avançado (Issue #10)
        advanced_model, target_map = train_advanced_model(preprocessor, X_train, y_train)
        
        print("\n--- Avaliação do Modelo Avançado (XGBoost Ponderado) ---")
        
        # O XGBoost prevê números (0, 1, 2), precisamos mapear de volta para string
        y_pred_advanced_mapped = advanced_model.predict(X_test)
        
        # Criar o mapa reverso (ex: {0: 'Empate', ...})
        target_map_reverse = {v: k for k, v in target_map.items()}
        y_pred_advanced = pd.Series(y_pred_advanced_mapped).map(target_map_reverse)
        
        accuracy_advanced = accuracy_score(y_test, y_pred_advanced)
        print(f"\nAcurácia (Avançado Ponderado): {accuracy_advanced * 100:.2f} %")
        print("\nRelatório de Classificação (Avançado Ponderado):")
        print(classification_report(y_test, y_pred_advanced))
        
        # 7. Comparação
        print("\n--- Comparação de Modelos Ponderados ---")
        print(f"Acurácia Baseline (Logistic): {accuracy_baseline * 100:.2f} %")
        print(f"Acurácia Avançado (XGBoost):  {accuracy_advanced * 100:.2f} %")