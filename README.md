# Projeto de Treinamento e Rastreamento com MLflow

## 1. Visão Geral do Projeto

Este repositorio fornece uma estrutura padronizada para treinar e rastrear experimentos de Machine Learning usando MLflow. Ele é projetado para ser modular, permitindo a fácil integração de diferentes tipos de modelos (ex: Detecção de Objetos, Classificação) através de "trainers" e arquivos de configuração YAML.

O objetivo principal é centralizar o rastreamento de todos os parâmetros, métricas e artefatos de nossos modelos para garantir reprodutibilidade e colaboração.

## 2. Estrutura de Arquivos

O projeto está organizado da seguinte forma:

```
.
├── configs/
│   ├── config_classificacao.yaml  # Configs para modelos de classificação
│   └── config_deteccao.yaml       # Configs para modelos de detecção
├── trainers/
│   ├── __init__.py
│   ├── detection_trainer.py       # Lógica de treino para detecção (YOLO)
│   └── generic_classification_trainer.py # Lógica para classificação (Scikit-learn)
├── train.py                         # Script principal para iniciar os treinos
└── README.md                        # Esta documentação
```

* **`train.py`:** O orquestrador principal. Ele lê um arquivo de configuração, inicia uma run no MLflow e chama o trainer apropriado.
* **`configs/`:** Contém os arquivos de configuração em formato YAML. Cada arquivo define um experimento completo: dados, parâmetros, métricas, etc.
* **`trainers/`:** Contém os "motores" de treinamento. Cada arquivo é especializado em uma tarefa (ex: `detection_trainer.py` sabe como treinar um modelo YOLO).

## 3. Pré-requisitos

Antes de executar um treinamento, garanta que você tem:
* Python 3.9+
* As bibliotecas Python listadas no arquivo `requirements.txt`.
* (Para treinos de detecção) Uma chave de API da Roboflow configurada no `config_deteccao.yaml`.

## 4. Como Executar um Experimento (Passo a Passo)

### Passo 1: Instalar as Dependências

Clone o projeto e instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### Passo 2: Executar o Script de Treinamento

Em outro terminal, na raiz do projeto, execute o script `train.py`, apontando para o arquivo de configuração desejado.

**Exemplo para um treino de detecção:**
```bash
python train.py --config configs/config_deteccao.yaml
```

**Exemplo para um treino de classificação:**
```bash
python train.py --config configs/config_classificacao.yaml
```

### Passo 3: Visualizar os Resultados

Abra seu navegador e acesse a interface do MLflow para ver sua run, comparar resultados e analisar os artefatos.


## 4. Adicionando um Novo Trainer

Para adicionar um novo tipo de modelo (ex: segmentação):
1.  Crie um novo arquivo `meu_novo_trainer.py` dentro da pasta `trainers/`.
2.  Implemente a função `run(config)` dentro dele, contendo a lógica de treino.
3.  Adicione um `elif` no `train.py` para chamar seu novo trainer quando o `trainer_type` corresponder.
4.  Crie um novo arquivo `config_meu_novo_modelo.yaml` na pasta `configs/`.