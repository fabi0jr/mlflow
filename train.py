import mlflow
import yaml
import argparse
import os
import shutil
import tempfile
from trainers import detection_trainer, generic_classification_trainer, image_classification_trainer

mlflow.set_tracking_uri("http://172.100.11.45:5000")

def main(config_path):
    
    temp_dir = tempfile.mkdtemp(prefix="mlflow_run_")
    print(f"Diretório de trabalho temporário isolado criado em: {temp_dir}")
    
    try:
        # 1. Carregar a configuração
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 2. Iniciar a run do MLflow
        experiment_name = config['experiment_name']
        
        # 2. Verificar se o experimento já existe
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # 2. Se NÃO existe, criamos com um caminho de artefato legível
            print(f"Experimento '{experiment_name}' não encontrado. Criando um novo...")
            artifact_location = f"s3://mlflow/{experiment_name}"
            try:
                mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                print(f"Novo experimento criado com local de artefato: {artifact_location}")
            except mlflow.exceptions.MlflowException as e:
                print(f"Erro ao criar experimento (pode já existir): {e}")
                pass
        
        elif experiment.lifecycle_stage == 'deleted':
            # 3. SE EXISTIR, MAS ESTIVER DELETADO
            print(f"ERRO: O experimento '{experiment_name}' existe, mas está na lixeira.")
            print("Por favor, acesse a UI do MLflow, vá para a aba 'Experiments',")
            print("e 'Restaure' o experimento ou 'Delete Permanentemente' para recriá-lo.")
            return
        
        # 4. Definir o experimento (agora sabemos que ele existe e está ativo)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=config['run_name']) as run:
            
            print(f"Iniciando run '{config['run_name']}' no experimento '{config['experiment_name']}'...")
            if 'params' in config:
                mlflow.log_params(config['params'])
            mlflow.log_dict(config, "config.yaml")

            # 3. Selecionar e executar o trainer
            if config['trainer_type'] == 'detection':
                
                results, data_yaml_path = detection_trainer.run(config, temp_dir)
                
                print(f"Logando artefato de entrada: {data_yaml_path}")
                mlflow.log_artifact(data_yaml_path, "Configuracoes de Entrada")
                
                # Logar Métricas
                if 'metrics_to_log' in config:
                    metric_name_map = {
                        'mAP_50': 'metrics/mAP50(B)', 'mAP_50_95': 'metrics/mAP50-95(B)',
                        'Precision': 'metrics/precision(B)', 'Recall': 'metrics/recall(B)',
                        'val_loss': 'val/box_loss', 'train_loss': 'train/box_loss',
                        'F1-Score': 'metrics/f1(B)'
                    }
                    metrics_to_log = {}
                    for metric_name in config['metrics_to_log']:
                        source_name = metric_name_map.get(metric_name)
                        if source_name:
                            metrics_to_log[metric_name] = results.results_dict.get(source_name, 0)
                    
                    mlflow.log_metrics(metrics_to_log)

                # Logar Artefatos de Saída
                if 'output_artifacts_to_log' in config:
                    print("Logando artefatos de saída do modelo...")
                    for name, path in config['output_artifacts_to_log'].items():
                        # results.save_dir agora é /tmp/mlflow_run.../yolo_results/run_name
                        full_path = os.path.join(results.save_dir, path)
                        if os.path.exists(full_path):
                            mlflow.log_artifact(full_path, f"Resultados do Treino/{name}")

            elif config['trainer_type'] == 'image_classification':
                
                results = image_classification_trainer.run(config, temp_dir)
                
                if 'metrics_to_log' in config:
                    metric_name_map = {
                        'train_loss': 'train/loss',
                        'val_loss': 'val/loss',
                        'top1_accuracy': 'metrics/accuracy_top1',
                        'top5_accuracy': 'metrics/accuracy_top5'
                    }
                    metrics_to_log = {}
                    for metric_name in config['metrics_to_log']:
                        source_name = metric_name_map.get(metric_name)
                        if source_name:
                            metrics_to_log[metric_name] = results.results_dict.get(source_name, 0)
                    
                    mlflow.log_metrics(metrics_to_log)

                if 'output_artifacts_to_log' in config:
                    print("Logando artefatos de saída do modelo...")
                    for name, path in config['output_artifacts_to_log'].items():
                        full_path = os.path.join(results.save_dir, path)
                        if os.path.exists(full_path):
                            mlflow.log_artifact(full_path, f"Resultados do Treino/{name}")
            
            print("Run finalizada com sucesso!")

    finally:
        if os.path.exists(temp_dir):
            print(f"Limpando diretório temporário: {temp_dir}")
            shutil.rmtree(temp_dir)
            print("Limpeza concluída. O servidor está limpo.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Caminho para o arquivo de configuração YAML.")
    args = parser.parse_args()
    
    main(args.config)