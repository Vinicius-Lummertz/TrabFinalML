import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer # <-- Adicionado FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Ignorar warnings de convergência da Regressão Logística, que são comuns
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---
# 1. Carregar e Processar Todos os Dados
# ---
from data_loader import load_data
from preprocessing import create_target_variable, clean_data
from feature_engineering import consolidate_match_data, create_rolling_features # <-- Use seu nome de arquivo se for 'engeneering'

def get_final_data():
    """
    Executa todo o pipeline de ETL (Extract, Transform, Load)
    para obter os dados prontos para o modelo.
    """
    print("ℹ️ Executando pipeline completo de ETL...")
    # Carga (Issue #3)
    df_full, df_stats, df_gols, df_cartoes = load_data()
    if df_full is None:
        print("❌ Falha no carregamento. Abortando.")
        return None
    
    # Limpeza e Alvo (Issues #4, #5)
    df_full = create_target_variable(df_full)
    df_full_clean, df_stats_clean = clean_data(df_full, df_stats)
    
    # Engenharia de Features (Issue #6)
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
    
    # **** CORREÇÃO AQUI ****
    # Helper function para converter colunas para string
    def to_string(X):
        return X.astype(str)

    categorical_transformer = Pipeline(steps=[
        ('tostring', FunctionTransformer(to_string)), # <-- Adicionado: Força para string
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
    
    print("✅ Preprocessor criado (com blindagem de tipo string).")
    return preprocessor

# ---
# 3. Treinar Modelo Baseline (Issue #9)
# ---
def train_baseline_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Cria e treina o modelo baseline (LogisticRegression) dentro de um Pipeline.
    """
    print("ℹ️ Treinando modelo Baseline (Regressão Logística) (Issue #9)...")
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    print("✅ Modelo Baseline treinado.")
    return model_pipeline


# --- Bloco de Teste ---
if __name__ == "__main__":
    print("--- Testando o módulo model_training.py ---")
    
    # 1. Obter dados
    df_modelo = get_final_data()
    
    if df_modelo is not None:
        # 2. Separar X e y
        y = df_modelo['resultado']
        X = df_modelo.drop(columns=['resultado', 'id'])
        
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
        
        # 5. Treinar Modelo Baseline (Issue #9)
        baseline_model = train_baseline_model(preprocessor, X_train, y_train)
        
        # 6. Avaliar Modelo Baseline
        print("\n--- Avaliação do Modelo Baseline (Regressão Logística) ---")
        y_pred = baseline_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAcurácia (Teste): {accuracy * 100:.2f} %")
        
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        print("\nMatriz de Confusão (Teste):")
        cm = confusion_matrix(y_test, y_pred, labels=baseline_model.classes_)
        cm_df = pd.DataFrame(cm, index=baseline_model.classes_, columns=baseline_model.classes_)
        print(cm_df)