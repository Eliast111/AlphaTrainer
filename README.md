# AlphaTrainer 1.3.0 — plataforma multimodelo de IA

O **AlphaTrainer** é um projeto educacional e experimental da Alpha Technologies para criar, treinar, executar, avaliar e comparar modelos de inteligência artificial. Inclui redes neuronais densas e um Transformer de linguagem, além de ferramentas de gestão e motores universais.

O AlphaTrainer é um produto independente da aplicação Alpha.

## Requisitos

- Python **3.14.x** (requisito declarado em `pyproject.toml`).
- Flet **1.1.2** para a aplicação Flet.
- NumPy, SciPy e Matplotlib podem ser utilizados como acelerações ou ferramentas opcionais, conforme o ambiente. Não são dependências obrigatórias declaradas em `pyproject.toml`.

A disponibilidade de interfaces e dependências pode variar conforme a plataforma. A configuração Android e a aplicação Web têm instruções próprias nos respetivos documentos.

## Início rápido

A partir da pasta do projeto:

```bash
python main.py
python main.py demo xor
python main.py demo seno
```

O menu interativo e os comandos disponíveis dependem da versão atual do ponto de entrada `main.py`.

## Funcionalidades principais

- **Redes neuronais densas:** camadas, funções de ativação, propagação direta e retropropagação.
- **Aprendizagem:** funções de perda, gradientes e otimizadores SGD, Momentum e Adam.
- **Treino:** mini-batches, validação, early stopping, clipping de gradientes e callbacks, conforme o treinador utilizado.
- **Transformer de linguagem:** tokenização, vocabulário, embeddings, codificação posicional, atenção causal, camadas Transformer e geração de texto.
- **Dados:** datasets, leitura CSV, validação e divisão de dados.
- **Avaliação:** métricas numéricas, classificação, relatórios e avaliação de modelos de linguagem.
- **Checkpoints e persistência:** mecanismos para guardar e recuperar modelos e dados suportados pelos módulos existentes.
- **Gestão multimodelo:** fábrica, contratos de capacidades, adaptadores, gestor de modelos e registo de plugins confiáveis.

## Motores universais

| Motor | Módulo | Função |
|---|---|---|
| Treino universal | `training/universal.py` | Encaminha o treino para o treinador adequado à arquitetura. |
| Avaliação universal | `evaluation/universal.py` | Avalia modelos e organiza métricas e detalhes. |
| Benchmark universal | `evaluation/benchmark.py` | Compara pelo menos dois modelos através de uma métrica comum. |
| Otimização universal | `optimization/universal.py` | Experimenta configurações de treino e compara resultados. |
| Runtime de IA | `ai_runtime/runtime.py` | Encaminha previsão, geração e operações `forward`. |

A palavra «universal» descreve a camada de encaminhamento e integração. Não significa que todas as arquiteturas ou operações de IA sejam suportadas. As capacidades dependem dos adaptadores e dos métodos realmente implementados por cada modelo.

## Interfaces

- **CLI:** ponto de entrada `main.py`.
- **Aplicação Flet:** `flet_app.py`, configurada em `pyproject.toml`.
- **Aplicação Web:** `webapp.py`; consultar [WEB.md](WEB.md).
- **Projeto Android:** pasta `android_app/`; consultar [ANDROID.md](ANDROID.md).

Para detalhes da estrutura interna, consultar [ARCHITECTURE.md](ARCHITECTURE.md). Para instruções específicas do Termux, consultar [TERMUX.md](TERMUX.md), se aplicável à tua instalação.

## Testes

Executar a partir da raiz do projeto:

```bash
python -m unittest discover -s tests -t .
```

Na validação isolada da versão atualizada da Fase 4, **440 testes passaram**. A validação final deve também ser executada no ambiente Termux/Android de destino.

## Limitações atuais

- O projeto é educativo e experimental; não deve ser considerado equivalente, em desempenho, escala ou maturidade, a frameworks como PyTorch ou TensorFlow.
- As arquiteturas suportadas dependem dos motores e adaptadores implementados e testados.
- O Transformer atual tem limitações arquiteturais; não deve ser apresentado como um modelo de linguagem de grande escala ou como um assistente conversacional pronto para produção.
- A aprovação dos testes não substitui testes de desempenho, segurança, compatibilidade entre dispositivos ou validação de aplicações distribuídas.

## Estado da Fase 4

A implementação da persistência versionada já integra checkpoints Dense e Transformer, metadados de ciclo de vida, manifesto V2, escrita atómica e leitura compatível com manifestos V1. A suite isolada passou com 440 testes; falta confirmar a mesma execução no ambiente Termux/Android de destino.
