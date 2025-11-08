import pandas as pd
import joblib
import json
from fastapi import FastAPI, HTTPException # <-- Importar HTTPException
from pydantic import BaseModel
from pathlib import Path

# --- IMPORTAÇÃO CORRIGIDA ---
# Importar a função utilitária usando import relativo dentro do pacote src.
# O unpickler (joblib.load) precisa que esta função esteja no escopo.
from .utils import to_string 

# ---
# 1. Setup da Aplicação e Carregamento do Modelo
# ---

# Encontrar o caminho raiz do projeto e a pasta de modelos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = PROJECT_ROOT / "models"

# Tentar carregar os artefatos (modelo e mapas)
try:
    # 1. Carregar o Pipeline do Modelo
    pipeline_path = MODELS_PATH / "brasileirao_xgb_pipeline.joblib"
    model_pipeline = joblib.load(pipeline_path)
    print(f"✅ Modelo pipeline carregado de {pipeline_path}")

    # 2. Carregar o Mapeamento de Target
    map_path = MODELS_PATH / "target_map.json"
    with open(map_path, 'r', encoding='utf-8') as f:
        target_map = json.load(f)
    
    # 3. Criar o Mapa Reverso (ex: {0: "Empate", 1: "Vitoria_Mandante", ...})
    #    necessário para retornar a string para o usuário
    target_map_reverse = {v: k for k, v in target_map.items()}
    print(f"✅ Mapeamento de target carregado: {target_map_reverse}")

except FileNotFoundError:
    print("❌ ERRO: Arquivos de modelo (brasileirao_xgb_pipeline.joblib) ou mapa (target_map.json) não encontrados.")
    print("Por favor, execute 'python src/model_training.py' primeiro para treinar e salvar os artefatos.")
    model_pipeline = None
    target_map_reverse = None
except Exception as e:
    print(f"❌ ERRO Inesperado ao carregar modelo: {e}")
    model_pipeline = None
    target_map_reverse = None

# Inicializar o app FastAPI
app = FastAPI(
    title="API de Previsão do Brasileirão",
    description="Uma API para prever resultados de partidas (Vitória Mandante, Empate, Vitória Visitante) usando um modelo XGBoost.",
    version="1.0.0"
)

# ---
# 2. Definir o Schema de Entrada (Pydantic BaseModel)
# ---

class PartidaInput(BaseModel):
    """
    Schema de dados para a entrada de uma nova partida.
    Baseado nas 16 features numéricas e 6 categóricas do log de treino.
    """
    
    # Categóricas (6)
    mandante: str         # Ex: "Flamengo"
    visitante: str        # Ex: "Palmeiras"
    arena: str            # Ex: "Maracanã"
    mandante_estado: str  # Ex: "RJ"
    visitante_estado: str # Ex: "SP"
    turno: str            # Ex: "Tarde", "Noite", "Manha"
    
    # Numéricas (16)
    rodata: int                                 # Ex: 10
    visitante_cartao_amarelo: int               # Esta feature foi identificada no seu log de treino
    mandante_media_pontos_ult_5: float          # Ex: 1.8
    mandante_media_gols_feitos_ult_5: float     # Ex: 1.2
    mandante_media_gols_sofridos_ult_5: float   # Ex: 0.8
    mandante_media_chutes_ult_5: float          # Ex: 12.5
    mandante_media_chutes_no_alvo_ult_5: float  # Ex: 4.2
    mandante_media_posse_de_bola_ult_5: float   # Ex: 55.0
    mandante_media_faltas_ult_5: float          # Ex: 15.0
    visitante_media_pontos_ult_5: float         # Ex: 1.4
    visitante_media_gols_feitos_ult_5: float    # Ex: 1.0
    visitante_media_gols_sofridos_ult_5: float  # Ex: 1.1
    visitante_media_chutes_ult_5: float         # Ex: 10.3
    visitante_media_chutes_no_alvo_ult_5: float # Ex: 3.8
    visitante_media_posse_de_bola_ult_5: float  # Ex: 48.5
    visitante_media_faltas_ult_5: float         # Ex: 17.2

    # Exemplo de como enviar os dados (JSON)
    class Config:
        json_schema_extra = {
            "example": {
                "mandante": "Flamengo", "visitante": "Vasco", "arena": "Maracanã",
                "mandante_estado": "RJ", "visitante_estado": "RJ", "turno": "Noite",
                "rodata": 15, "visitante_cartao_amarelo": 2,
                "mandante_media_pontos_ult_5": 2.2, "mandante_media_gols_feitos_ult_5": 1.8,
                "mandante_media_gols_sofridos_ult_5": 0.5, "mandante_media_chutes_ult_5": 14.0,
                "mandante_media_chutes_no_alvo_ult_5": 5.1, "mandante_media_posse_de_bola_ult_5": 60.1,
                "mandante_media_faltas_ult_5": 12.0,
                "visitante_media_pontos_ult_5": 1.0, "visitante_media_gols_feitos_ult_5": 0.8,
                "visitante_media_gols_sofridos_ult_5": 1.5, "visitante_media_chutes_ult_5": 9.0,
                "visitante_media_chutes_no_alvo_ult_5": 2.5, "visitante_media_posse_de_bola_ult_5": 40.5,
                "visitante_media_faltas_ult_5": 16.5
            }
        }


class PredictionOutput(BaseModel):
    """Schema de saída da previsão."""
    previsao_numerica: int      # Ex: 0
    previsao_resultado: str     # Ex: "Empate"

# ---
# 3. Definir Endpoints da API
# ---

@app.get("/", summary="Endpoint raiz para verificar o status da API")
def read_root():
    """
    Verifica se a API está online e se o modelo foi carregado.
    """
    return {
        "status": "online",
        "modelo_carregado": model_pipeline is not None
    }

@app.post("/predict", summary="Realiza a previsão de uma partida", response_model=PredictionOutput)
def predict(partida: PartidaInput):
    """
    Recebe os dados de uma partida e retorna a previsão do resultado.
    
    **Resultados Possíveis:**
    - `Vitoria_Mandante`
    - `Vitoria_Visitante`
    - `Empate`
    """
    if model_pipeline is None or target_map_reverse is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Modelo não carregado. Verifique os logs do servidor."
        )

    try:
        # 1. Converter a entrada Pydantic (1 partida) para um DataFrame
        input_data = pd.DataFrame([partida.model_dump()])
        
        # 2. Realizar a predição
        previsao_numerica = model_pipeline.predict(input_data)[0]
        
        # 3. Converter a previsão numérica (ex: 0) para string (ex: "Empate")
        previsao_num_int = int(previsao_numerica)
        previsao_resultado_str = target_map_reverse.get(previsao_num_int, "Classe Desconhecida")
        
        # 4. Retornar o JSON de saída
        return {
            "previsao_numerica": previsao_num_int,
            "previsao_resultado": previsao_resultado_str
        }
    except Exception as e:
        # Captura erros durante a previsão (ex: dados de entrada inesperados)
        raise HTTPException(
            status_code=400, # Bad Request
            detail=f"Erro durante a predição: {e}"
        )

# ---
# 4. Bloco para rodar o servidor (para testes)
# ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)