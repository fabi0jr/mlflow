import mlflow
import argparse
import os

mlflow.set_tracking_uri("http://172.100.11.45:5000") 

def main(args):
    # 1. Validar se o caminho local existe
    if not os.path.exists(args.local_path):
        print(f"Erro: Caminho local não encontrado: {args.local_path}")
        print("Por favor, verifique o caminho e tente novamente.")
        return

    experiment_name = args.experiment_name
    
    # 2. Verificar se o experimento já existe
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # 3. Se NÃO existe, criamos com um caminho de artefato legível
        print(f"Experimento '{experiment_name}' não encontrado. Criando um novo...")
        
        # Construímos o caminho: s3://<bucket_name>/<experiment_name>
        artifact_location = f"s3://mlflow/{experiment_name}"
        
        try:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            print(f"Novo experimento criado com local de artefato: {artifact_location}")
        except mlflow.exceptions.MlflowException as e:
            print(f"Erro ao criar experimento (pode já existir): {e}")
            pass
    
    elif experiment.lifecycle_stage == 'deleted':
        # 4. SE EXISTIR, MAS ESTIVER DELETADO (Proteção)
        print(f"ERRO: O experimento '{experiment_name}' existe, mas está na lixeira.")
        print("Por favor, acesse a UI do MLflow e 'Delete Permanentemente' para recriá-lo.")
        return

    # 5. Definir o experimento
    mlflow.set_experiment(experiment_name)

    # 3. Iniciar a run do MLflow
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"Iniciando run '{args.run_name}' (ID: {run_id}) no experimento '{args.experiment_name}'...")

        # 4. Logar parâmetros e descrição
        # Isso é ótimo para rastrear metadados
        mlflow.log_param("local_path_source", args.local_path)
        if args.description:
            mlflow.log_param("description", args.description)
            # Também podemos salvar a descrição como uma tag
            mlflow.set_tag("description", args.description)

        # 5. O passo mais importante: Logar o artefato (o dataset)
        print(f"Fazendo upload do dataset de '{args.local_path}' para o artifact path '{args.artifact_path}'...")
        
        try:
            mlflow.log_artifact(
                local_path=args.local_path,
                artifact_path=args.artifact_path
            )
        except Exception as e:
            print(f"Erro durante o upload do artefato: {e}")
            return

        # 6. Sucesso!
        print("\n--- SUCESSO! ---")
        print(f"Dataset registrado com sucesso no MLflow.")
        print(f"Acesse a UI do MLflow para ver o experimento '{args.experiment_name}'.")
        print(f"\nUse este ID de Run no seu arquivo .yaml de treino:")
        print(f"=====================================")
        print(f"  dataset_run_id: \"{run_id}\"")
        print(f"=====================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Registra um dataset local (pasta ou arquivo) no Servidor MLflow Central.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--local_path",
        required=True,
        help="O caminho para a pasta (ex: ./dataset_xadrez) ou arquivo (ex: ./iris.csv) do dataset."
    )
    
    parser.add_argument(
        "--run_name",
        required=True,
        help="O nome que este dataset terá no MLflow (ex: 'dataset_xadrez_v1')."
    )
    
    parser.add_argument(
        "--artifact_path",
        required=True,
        help="O nome da 'pasta' DENTRO da run do MLflow onde os dados serão salvos.\n"
             "Exemplo para detecção: 'dataset' (assim o data.yaml fica em 'dataset/data.yaml').\n"
             "Exemplo para classificação: 'data' (assim o iris.csv fica em 'data/iris.csv')."
    )

    parser.add_argument(
        "--description",
        help="Uma descrição opcional sobre o dataset (ex: 'Versão inicial do dataset de xadrez com 12 classes')."
    )
    
    parser.add_argument(
        "--experiment_name",
        default="Datasets",
        help="O nome do experimento no MLflow para agrupar os datasets. O padrão é 'Datasets'."
    )

    args = parser.parse_args()
    main(args)