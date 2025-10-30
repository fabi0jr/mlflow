import mlflow
import os
import zipfile
from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def get_data_yaml_path(data_config, base_download_dir):
    """
    Decide de onde carregar o dataset (Roboflow, Local ou Servidor MLflow)
    e retorna o caminho para o data.yaml.
    """
    
    # Modo 1: Roboflow
    if 'roboflow_workspace' in data_config:
        print(">>> Baixando o dataset do Roboflow...")
        rf = Roboflow(api_key=data_config.get('roboflow_api_key'))
        project = rf.workspace(data_config['roboflow_workspace']).project(data_config['roboflow_project'])
        version = project.version(data_config['roboflow_version'])
        dataset = version.download(data_config.get('download_format', 'yolov8'))
        return dataset.location + "/data.yaml"

    elif 'local_data_yaml' in data_config:
        print(f">>> Usando dataset local em: {data_config['local_data_yaml']}")
        return data_config['local_data_yaml']
        
    elif 'dataset_run_id' in data_config:
        run_id = data_config['dataset_run_id']
        artifact_zip_path = data_config['dataset_artifact_path']
        data_yaml_relative = data_config['data_yaml_relative_path']
        
        print(f">>> Baixando dataset do servidor MLflow (Run ID: {run_id})...")
        
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

        # 2. Descompactar o arquivo no mesmo diretório
        unzip_dir = os.path.join(base_download_dir, "unzipped_data")
        print(f"Descompactando para: {unzip_dir}...")
        
        with zipfile.ZipFile(downloaded_zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            
        print("Descompactação concluída.")

        # 3. Retornar o caminho final para o data.yaml
        final_data_yaml_path = os.path.join(unzip_dir, data_yaml_relative)
        
        if not os.path.exists(final_data_yaml_path):
            print(f"ERRO: O 'data_yaml_relative_path' está incorreto!")
            raise FileNotFoundError(final_data_yaml_path)
            
        return final_data_yaml_path
        
    else:
        raise ValueError("Configuração de 'data' inválida.")

def run(config, temp_dir):
    """
    Executa o treinamento, traduzindo os parâmetros genéricos da config
    para os nomes específicos que o YOLO espera.
    """
    print("--- Executando o Trainer de Detecção (com Fontes de Dados Flexíveis) ---")

    data_config = config['data']
    params_config = config['params']
    
    data_yaml_path = get_data_yaml_path(data_config, temp_dir)
    
    SETTINGS.update({'mlflow': False})
    model = YOLO(params_config['model_name'])
    
    yolo_params = {
        'imgsz': params_config.get('image_size'),
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

    print(f"Iniciando treinamento com data: {data_yaml_path}")
    results = model.train(
        data=data_yaml_path,
        project=os.path.join(temp_dir, "yolo_results"),
        name=config['run_name'],
        **yolo_params 
    )
    
    return results, data_yaml_path