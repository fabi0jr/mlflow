import mlflow
import pandas as pd
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_data_path(data_config, base_download_dir):
    """
    Decide de onde carregar o dataset (Local ou Servidor MLflow)
    e retorna o caminho para o arquivo .csv.
    """
    
    # Modo 1: Local (caminho direto)
    if 'path' in data_config:
        print(f">>> Usando dataset local em: {data_config['path']}")
        return data_config['path']
        
    # Modo 2: Servidor MLflow (Dataset .csv ou .zip)
    elif 'dataset_run_id' in data_config:
        run_id = data_config['dataset_run_id']
        artifact_path = data_config['dataset_artifact_path']
        
        print(f">>> Baixando dataset do servidor MLflow (Run ID: {run_id})...")
        
        try:
            downloaded_file_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=base_download_dir
            )
        except Exception as e:
            print(f"Erro ao baixar artefato: {e}")
            raise
            
        print(f"Artefato baixado para: {downloaded_file_path}")

        if downloaded_file_path.endswith(".zip"):
            unzip_dir = os.path.join(base_download_dir, "unzipped_data")
            print(f"Descompactando para: {unzip_dir}...")
            
            with zipfile.ZipFile(downloaded_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            
            print("Descompactação concluída.")
            
            data_file_relative = data_config.get('data_file_relative_path')
            if not data_file_relative:
                raise ValueError("Config 'data_file_relative_path' é necessária ao usar um .zip")
                
            final_data_path = os.path.join(unzip_dir, data_file_relative)
        
        else:
            final_data_path = downloaded_file_path

        if not os.path.exists(final_data_path):
            raise FileNotFoundError(f"Arquivo de dados não encontrado em: {final_data_path}")
            
        return final_data_path
        
    else:
        raise ValueError("Configuração de 'data' inválida. Especifique 'path' ou 'dataset_run_id'.")


def run(config, temp_dir):
    """
    Executa um treinamento de classificação genérico.
    Agora usa um temp_dir para isolamento.
    """
    print("--- Executando o Trainer de Classificação (Modo Seguro para Equipe) ---")
    
    # 1. Carregar dados (com a nova lógica)
    data_path = get_data_path(config['data'], temp_dir) 
    
    target_col = config['data']['target_column']
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config['params']['seed'])
    
    # 2. Treinamento do modelo
    p = config['params']
    model_hyperparams = {"n_estimators": p['n_estimators'], "max_depth": p['max_depth'], "random_state": p['seed']}
    model = RandomForestClassifier(**model_hyperparams)
    model.fit(X_train, y_train)
    
    # 3. Calcular predições para as métricas
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)

    # 4. Registrar métricas e modelo (isso pode ficar aqui, é simples)
    print("Calculando e registrando métricas...")
    
    metrics_to_log = {
        "Top-1 Accuracy": accuracy_score(y_test, preds),                   
        "Precision": precision_score(y_test, preds, average="weighted"),   
        "Recall": recall_score(y_test, preds, average="weighted"),         
        "F1-Score": f1_score(y_test, preds, average="weighted"),           
        "ROC-AUC": roc_auc_score(y_test, preds_proba, multi_class='ovr', average="weighted") 
    }

    print(f"Métricas calculadas: {metrics_to_log}")
    mlflow.log_metrics(metrics_to_log)
    
    mlflow.sklearn.log_model(sk_model=model, name=p['model_name'], input_example=X_train)
    
    print("Treinamento genérico concluído.")
    return model