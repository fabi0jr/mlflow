# trainers/generic_classification_trainer.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def run(config):
    """
    Executa um treinamento de classificaÃ§Ã£o genÃ©rico a partir de um arquivo de dados.
    """
    print("--- Executando o Trainer de ClassificaÃ§Ã£o GenÃ©rico (com MÃ©tricas Completas) ---")
    
    # 1. Carregar dados
    data_path = config['data']['path']
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
    
    # 3. Calcular prediÃ§Ãµes para as mÃ©tricas
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)

    # 4. Registrar todas as mÃ©tricas de classificaÃ§Ã£o relevantes
    print("Calculando e registrando mÃ©tricas...")
    
    metrics_to_log = {
        "Top-1 Accuracy": accuracy_score(y_test, preds),                   
        "Precision": precision_score(y_test, preds, average="weighted"),   
        "Recall": recall_score(y_test, preds, average="weighted"),         
        "F1-Score": f1_score(y_test, preds, average="weighted"),           
        "ROC-AUC": roc_auc_score(y_test, preds_proba, multi_class='ovr', average="weighted") 
    }

    print(f"MÃ©tricas calculadas: {metrics_to_log}")
    mlflow.log_metrics(metrics_to_log)
    mlflow.sklearn.log_model(sk_model=model, name=p['model_name'], input_example=X_train)
    
    print("Treinamento genÃ©rico concluÃ­do.")
    return model