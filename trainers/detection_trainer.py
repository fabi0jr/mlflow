# trainers/detection_trainer.py

from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def run(config):
    """
    Executa o treinamento, traduzindo os parÃ¢metros genÃ©ricos da config
    para os nomes especÃ­ficos que o YOLO espera.
    """
    print("--- Executando o Trainer de DetecÃ§Ã£o (com TraduÃ§Ã£o de ParÃ¢metros) ---")
    
    data_config = config['data']
    params_config = config['params']
    
    # Baixar dataset
    rf = Roboflow(api_key=data_config['roboflow_api_key'])
    project = rf.workspace(data_config['roboflow_workspace']).project(data_config['roboflow_project'])
    version = project.version(data_config['roboflow_version'])

    print(">>> Baixando o dataset do Roboflow... Isso pode levar alguns minutos na primeira vez. Por favor, aguarde.")

    dataset = version.download(data_config.get('download_format', 'yolov8'))
    data_yaml_path = dataset.location + "/data.yaml"
    
    SETTINGS.update({'mlflow': False})
    model = YOLO(params_config['model_name'])
    
    # --- CAMADA DE TRADUÇÃO ---

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
    # Removemos chaves que nÃ£o foram definidas para nÃ£o passar 'None'
    yolo_params = {k: v for k, v in yolo_params.items() if v is not None}

    results = model.train(
        data=data_yaml_path,
        name=config['run_name'],
        **yolo_params # Passamos o dicionÃ¡rio traduzido
    )
    
    return results, data_yaml_path