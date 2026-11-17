# AlphaTrainer 1.0

### Alpha Technologies

AlphaTrainer e um framework educacional de inteligencia artificial desenvolvido em Python.

## Funcionalidades

- Motor matematico e tensores.
- Redes neuronais e camadas.
- Backpropagation e gradientes.
- Otimizadores e treinamento.
- Gestao de datasets.
- Avaliacao e comparacao de modelos.
- AI Model Factory extensivel e Model Registry.
- Runtime nativo para inferencia e geracao de tokens, sem dependencias externas.
- Checkpoints e persistencia.
- Tokenizacao, vocabulario e embeddings.
- Arquitetura Transformer.
- Interface interativa de terminal.

## Estrutura

- alpha_math/ — motor matematico.
- neural/ — redes neuronais.
- learning/ — aprendizagem.
- training/ — treinamento.
- data/ — datasets.
- evaluation/ — avaliacao e relatorios.
- models/ — criacao de modelos e AI Factory.
- registry/ — registo de modelos.
- runtime/ — inferencia e geracao com modelos nativos.
- checkpoints/ — checkpoints.
- persistence/ — persistencia.
- language/ — processamento de linguagem.
- transformer/ — arquitetura Transformer.
- tests/ — testes automaticos.
- main.py — interface principal.

## Requisitos

- Python 3.10 ou superior.
- Terminal compativel com Python.
- Sem dependencias externas obrigatorias.

## Executar

    python main.py

## Menu principal

1. Criar modelo
2. Listar modelos
3. Remover modelo
4. Estado do sistema
5. Treinar modelo
6. Avaliar modelo
7. Gerar relatorio
8. Comparar modelos
0. Guardar dados e sair

## Factory e runtime

A factory unificada cria redes feedforward e modelos de linguagem. Tipos adicionais
podem ser registrados com um builder chamável:

    from models.ai_factory import AIModelFactory

    factory = AIModelFactory()
    model = factory.create("feedforward", layer_sizes=[2, 3, 1])

O runtime permite executar inferência em modelos com `forward()` e gerar tokens
para modelos de linguagem. A interface de alto nível integra criação, inferência
e geração:

    from core import AlphaCore
    from language.vocabulary import Vocabulary

    alpha = AlphaCore()
    vocabulary = Vocabulary().build([["ola", "mundo"]])
    alpha.create_language_model("texto", vocabulary)
    resultado = alpha.generate_text("texto", "ola", max_new_tokens=5)

`AlphaCore` também oferece `create_model_by_type()`, `run_model()`,
`predict_next()`, `generate_tokens()`, treinamento, avaliação e relatórios.
PyTorch e TensorFlow não são necessários.

## Persistencia

O PersistenceManager guarda e recupera modelos e datasets em persistence/saved/.

## Testes

Executar todos os testes:

    python -m unittest discover -v

## Objetivo

Construir uma base modular para estudar e desenvolver tecnologias proprias de inteligencia artificial, redes neuronais e modelos de linguagem.

---

Alpha Technologies — AlphaTrainer 1.0
