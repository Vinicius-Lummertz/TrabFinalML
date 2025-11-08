import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight 
import warnings
import numpy as np
import joblib 
import json   
from pathlib import Path 

# --- IMPORTAÇÃO CORRIGIDA ---
# Prefer package import, but fallback to top-level `utils` so joblib unpickler (which may
# reference `utils.to_string`) can find the symbol when loading artifacts.
try:
    from src.utils import to_string
except Exception:
    # When the saved model references `utils.to_string`, the unpickler will import `utils`.
    # Provide a fallback to the top-level module if available.
    from utils import to_string  # type: ignore

# Ignorar warnings de convergência
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---
# 1. Definir Caminhos
# ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(exist_ok=True) # Garantir que a pasta exista

# ---
# 2. Carregar e Processar Todos os Dados
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
# 3. Definir Pipeline de Pré-processamento (Issue #8)
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
    
    # *** CORREÇÃO AQUI: Usamos a função to_string importada ***
    categorical_transformer = Pipeline(steps=[
        ('tostring', FunctionTransformer(to_string, validate=False)), 
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
# 4. Treinar Modelos (Issues #9 e #11)
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
    
    target_map = {'Empate': 0, 'Vitoria_Mandante': 1, 'Vitoria_Visitante': 2}
    y_train_mapped = y_train.map(target_map)
    
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train_mapped
    )
    
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='multi:softmax', 
            num_class=3,             
            random_state=42,
            eval_metric='mlogloss' 
        ))
    ])
    
    # Usando os parâmetros exatos que você encontrou (para ser mais rápido)
    param_grid = {
        'classifier__learning_rate': [0.05], 
        'classifier__max_depth': [7], 
        'classifier__n_estimators': [200]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        scoring='f1_weighted', 
        cv=3,                  
        n_jobs=-1,             
        verbose=1              
    )
    
    fit_params = {
        'classifier__sample_weight': sample_weights
    }
    
    grid_search.fit(X_train, y_train_mapped, **fit_params)
    
    print("✅ Otimização (GridSearch) concluída.")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    
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
        
        # ------------------------------------------------------------------
        # --- ETAPA 6: SALVAR O MODELO (Issue #12) ---
        # ------------------------------------------------------------------
        print("\nℹ️ Serializando o melhor modelo (Issue #12)...")
        
        # 1. Salvar o pipeline completo (pré-processador + classificador)
        pipeline_path = MODELS_PATH / "brasileirao_xgb_pipeline.joblib"
        joblib.dump(best_advanced_model, pipeline_path)
        
        # 2. Salvar o mapeamento de classes (para a API saber {0: 'Empate', ...})
        map_path = MODELS_PATH / "target_map.json"
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(target_map, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Pipeline salvo em: {pipeline_path}")
        print(f"✅ Mapeamento de alvo salvo em: {map_path}")