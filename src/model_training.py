import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # <-- Importar GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight 
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
# 3. Treinar Modelos (Issues #9 e #11)
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
            class_weight='balanced' 
        ))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    print("✅ Modelo Baseline Ponderado treinado.")
    return model_pipeline

def tune_advanced_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Otimiza (TUNE) o modelo avançado (XGBoost) com GridSearchCV e ponderação.
    (Issue #11)
    """
    print("\nℹ️ Otimizando modelo Avançado (XGBoost Ponderado com GridSearchCV) (Issue #11)...")
    print("⚠️ Isso pode demorar alguns minutos...")
    
    # O XGBoost precisa que os alvos (y) sejam numéricos (0, 1, 2)
    target_map = {'Empate': 0, 'Vitoria_Mandante': 1, 'Vitoria_Visitante': 2}
    y_train_mapped = y_train.map(target_map)
    
    # Calcular pesos para o XGBoost
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train_mapped
    )
    
    # 1. Criar o Pipeline (igual antes)
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='multi:softmax', 
            num_class=3,             
            random_state=42,
            eval_metric='mlogloss' # Métrica de avaliação interna
        ))
    ])
    
    # 2. Definir a Grade de Parâmetros para testar
    # (O prefixo 'classifier__' diz ao pipeline para passar
    #  o parâmetro para o passo 'classifier')
    param_grid = {
        'classifier__n_estimators': [100, 200, 300], # Número de árvores
        'classifier__max_depth': [3, 5, 7],      # Profundidade das árvores
        'classifier__learning_rate': [0.1, 0.05]   # Taxa de aprendizado
    }
    
    # 3. Configurar o GridSearchCV
    # Usamos 'f1_weighted' como 'scoring' pois estamos desbalanceados
    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        scoring='f1_weighted', # Métrica de otimização
        cv=3,                  # 3-fold cross-validation
        n_jobs=-1,             # Usar todos os processadores
        verbose=1              # Mostrar progresso
    )
    
    # 4. Criar o dicionário de fit_params
    # (Para passar os sample_weights para o XGBoost DENTRO do gridsearch)
    fit_params = {
        'classifier__sample_weight': sample_weights
    }
    
    # 5. Executar a Otimização
    grid_search.fit(X_train, y_train_mapped, **fit_params)
    
    print("✅ Otimização (GridSearch) concluída.")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    
    # Retorna o MELHOR modelo encontrado
    return grid_search.best_estimator_, target_map


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

        # 6. Otimizar e Avaliar Modelo Avançado (Issue #11)
        best_advanced_model, target_map = tune_advanced_model(preprocessor, X_train, y_train)
        
        print("\n--- Avaliação do Modelo Avançado (XGBoost OTIMIZADO) ---")
        
        y_pred_advanced_mapped = best_advanced_model.predict(X_test)
        
        target_map_reverse = {v: k for k, v in target_map.items()}
        y_pred_advanced = pd.Series(y_pred_advanced_mapped).map(target_map_reverse)
        
        accuracy_advanced = accuracy_score(y_test, y_pred_advanced)
        print(f"\nAcurácia (Avançado Otimizado): {accuracy_advanced * 100:.2f} %")
        print("\nRelatório de Classificação (Avançado Otimizado):")
        print(classification_report(y_test, y_pred_advanced))
        
        # 7. Comparação
        print("\n--- Comparação de Modelos Ponderados ---")
        print(f"Acurácia Baseline (Logistic): {accuracy_baseline * 100:.2f} %")
        print(f"Acurácia Avançado (XGBoost Otimizado):  {accuracy_advanced * 100:.2f} %")
        print(f"Melhoria: {(accuracy_advanced - accuracy_baseline) * 100:.2f} pontos percentuais")