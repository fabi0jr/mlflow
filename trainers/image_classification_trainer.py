# trainers/image_classification_trainer.py
import mlflow
import os
import zipfile
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def get_data_path(data_config, base_download_dir):
    """
    Decide de onde carregar o dataset (Local ou Servidor MLflow)
    e retorna o caminho para a PASTA RAIZ do dataset (ex: /.../Bone-Break-Classification-2).
    """
    
    # Modo 1: Local (caminho direto)
    if 'data_root_path' in data_config:
        print(f">>> Usando dataset local em: {data_config['data_root_path']}")
        return data_config['data_root_path']
        
    # Modo 2: Servidor MLflow (Dataset .zip)
    elif 'dataset_run_id' in data_config:
        run_id = data_config['dataset_run_id']
        artifact_zip_path = data_config['dataset_artifact_path'] # Caminho para o .zip
        
        print(f">>> Baixando dataset de classificação do servidor (Run ID: {run_id})...")
        
        try:
            downloaded_zip_file = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_zip_path,
                dst_path=base_download_dir
            )
        except Exception as e:
            print(f"Erro ao baixar artefato: {e}")
            raise
            
        print(f"Dataset .zip baixado para: {downloaded_zip_file}")

        # Descompactar o arquivo no mesmo diretório
        unzip_dir = os.path.join(base_download_dir, "unzipped_data")
        print(f"Descompactando para: {unzip_dir}...")
        
        with zipfile.ZipFile(downloaded_zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            
        print("Descompactação concluída.")
        
        # O 'data_root_relative_path' é o nome da pasta dentro do zip
        # Ex: "Bone-Break-Classification-2"
        data_root_relative = data_config.get('data_root_relative_path')
        if not data_root_relative:
             raise ValueError("Config 'data_root_relative_path' é necessária ao usar um .zip para classificação")
        
        final_data_path = os.path.join(unzip_dir, data_root_relative)

        if not os.path.exists(final_data_path):
            raise FileNotFoundError(f"Pasta raiz do dataset não encontrada em: {final_data_path}")
            
        return final_data_path
        
    else:
        raise ValueError("Configuração de 'data' inválida. Especifique 'data_root_path' ou 'dataset_run_id'.")


def run(config, temp_dir):
    """
    Executa um treinamento de CLASSIFICAÇÃO DE IMAGENS genérico (YOLO-CLS).
    Usa o temp_dir fornecido para isolamento.
    """
    print("--- Executando o Trainer de Classificação de Imagens (Modo Seguro para Equipe) ---")
    
    data_config = config['data']
    params_config = config['params']
    
    # 1. Obter o caminho para a pasta raiz do dataset
    data_root_path = get_data_path(data_config, temp_dir)

    # 3. Carregar o modelo de CLASSIFICAÇÃO
    model_path = params_config['model_name'] 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo de classificação não encontrado no cache: {model_path}")

    model = YOLO(model_path)
    
    # 4. Carregar hiperparâmetros (YOLO-CLS usa os mesmos nomes do 'train')
    yolo_params = {
        'imgsz': params_config.get('image_size', 224),
        'batch': params_config.get('batch_size'),
        'epochs': params_config.get('epochs'),
        'optimizer': params_config.get('optimizer'),
        'lr0': params_config.get('lr0'),
        'lrf': params_config.get('lrf'),
        'momentum': params_config.get('momentum'),
        'weight_decay': params_config.get('weight_decay'),
        'warmup_epochs': params_config.get('warmup_epochs'),
        'warmup_momentum': params_config.get('warmup_momentum'),
        'warmup_bias_lr': params_config.get('warmup_bias_lr'),
        'dropout': params_config.get('dropout'),
        'seed': params_config.get('seed'),
    }
    yolo_params = {k: v for k, v in yolo_params.items() if v is not None}

    print(f"Iniciando treinamento de CLASSIFICAÇÃO com data em: {data_root_path}")
    
    # 5. Chamar o 'train'. O YOLO-CLS entende a estrutura de pastas.
    results = model.train(
        data=data_root_path,
        project=os.path.join(temp_dir, "yolo_results"), 
        name=config['run_name'],                         
        **yolo_params 
    )
    
    # 6. Retornar os resultados. 
    return results