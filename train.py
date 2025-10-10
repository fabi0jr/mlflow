# train.py (VERSÃO FINAL E CORRIGIDA)
import mlflow
import yaml
import argparse
import os

# Importa os nossos "motores" de treinamento
from trainers import detection_trainer, generic_classification_trainer

mlflow.set_tracking_uri("http://localhost:5000")

def main(config_path):
    # 1. Carregar a configuração do arquivo YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Iniciar a run do MLflow
    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run(run_name=config['run_name']):
        
        print(f"Iniciando run '{config['run_name']}' no experimento '{config['experiment_name']}'...")
        if 'params' in config:
            mlflow.log_params(config['params'])
        mlflow.log_dict(config, "config.yaml")

        # 3. Selecionar e executar o trainer correto
        if config['trainer_type'] == 'detection':
            results, data_yaml_path = detection_trainer.run(config)
            
            mlflow.log_artifact(data_yaml_path, "Configuracoes de Entrada")
            
            # Usando a lista 'metrics_to_log' do config mestre
            if 'metrics_to_log' in config:
                # Dicionário de tradução dos nomes do YOLO para os nossos nomes
                metric_name_map = {
                    'mAP_50': 'metrics/mAP50(B)', 'mAP_50_95': 'metrics/mAP50-95(B)',
                    'Precision': 'metrics/precision(B)', 'Recall': 'metrics/recall(B)',
                    'val_loss': 'val/box_loss', 'train_loss': 'train/box_loss',
                    'F1-Score': 'metrics/f1(B)' # YOLO pode não gerar F1 por padrão
                }
                
                metrics_to_log = {}
                for metric_name in config['metrics_to_log']:
                    source_name = metric_name_map.get(metric_name)
                    if source_name:
                        metrics_to_log[metric_name] = results.results_dict.get(source_name, 0)
                
                mlflow.log_metrics(metrics_to_log)

            if 'output_artifacts_to_log' in config:
                for name, path in config['output_artifacts_to_log'].items():
                    full_path = os.path.join(results.save_dir, path)
                    if os.path.exists(full_path):
                        mlflow.log_artifact(full_path, f"Resultados do Treino/{name}")

        elif config['trainer_type'] == 'generic_classification':
            generic_classification_trainer.run(config)
        
        print("Run finalizada com sucesso!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Caminho para o arquivo de configuração YAML.")
    args = parser.parse_args()
    
    main(args.config)