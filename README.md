# AlphaTrainer 1.2

Biblioteca e aplicação de **redes neuronais e Transformer em Python**, com
aceleração opcional por NumPy/SciPy e gráficos com Matplotlib. O motor mantém
uma alternativa Python puro para funcionar em ambientes sem essas bibliotecas.

Requer **Python 3.8+**. NumPy, SciPy e Matplotlib são opcionais; Pandas não é necessário.

## Início rápido

```bash
python main.py                 # menu interativo
python main.py demo xor        # treina uma rede a aprender XOR
python main.py demo seno       # regressão: aprende sin(x)
python main.py lm-train --text examples/corpus_pt.txt --epochs 80 \
       --prompt "o gato" --save saved_models/lm.json
python main.py generate --model saved_models/lm.json --prompt "o cao"
```

### Treinar as tuas próprias IAs

**Regressão** (prever números) — CSV com as colunas de entrada e, no fim, os alvos:

```bash
python main.py train --data casas.csv --layers 2,8,1 --epochs 500 \
       --scale standard --scale-targets --validation 0.2 --patience 40 \
       --save saved_models/casas.json
python main.py predict --model saved_models/casas.json --input 120,3
```

**Classificação** (prever categorias) — CSV com as entradas e, na **última coluna**,
o rótulo (texto ou número). A última camada tem de ter tantos neurónios
quantas as classes:

```bash
python main.py train --data examples/flores.csv --task classification \
       --layers 2,6,3 --epochs 300 --lr 0.03 --save saved_models/flores.json
python main.py predict --model saved_models/flores.json --input 7.4,3.0
# Classe prevista: grande  (99.7%)
```

A normalização (`--scale standard|minmax`) é ajustada só nos dados de treino e
guardada dentro do modelo, por isso o `predict` aceita valores nas unidades
originais. Em classificação mostra-se acurácia, matriz de confusão e
precisão/recall por classe.

### Comandos

| Comando | Para que serve |
|---|---|
| `demo [xor\|and\|or\|seno\|linear\|blobs]` | treina um exemplo pronto a usar |
| `train --data X --layers 2,4,1 ...` | treina uma rede com um CSV ou exemplo |
| `predict --model f.json --input 1,0` | usa uma rede guardada |
| `lm-train --text f.txt ...` | treina um modelo de linguagem (Transformer) |
| `generate --model f.json --prompt "..."` | gera texto |
| `info --model f.json` | mostra a arquitetura de um modelo guardado |
| `doctor` | verifica NumPy, SciPy e Matplotlib disponíveis |

Exemplo com os teus dados (CSV; as **últimas** colunas são os alvos):

```bash
python main.py train --data examples/housing.csv --layers 2,6,1 \
       --activation tanh --output-activation linear \
       --optimizer adam --lr 0.02 --epochs 400 --validation 0.25 \
       --patience 40 --save saved_models/casas.json
python main.py predict --model saved_models/casas.json --input 0.9,0.6
```

## Usar como biblioteca

```python
from core import AlphaCore
from data.samples import load_sample
from learning.optimizer import Adam

alpha = AlphaCore()
alpha.create_model("xor", [2, 4, 1], activation="tanh",
                   output_activation="sigmoid")
alpha.train_model("xor", load_sample("xor"), Adam(0.05),
                  epochs=500, batch_size=1, patience=50)
print(alpha.predict("xor", [1, 0]))
alpha.save_model("xor", "saved_models/xor.json")
```

Modelo de linguagem:

```python
from transformer import LanguageModel

texts = ["o gato dorme .", "o cao come ."]
model = LanguageModel.from_texts(texts, embedding_dim=16, hidden_dim=32)
model.fit(texts, epochs=60, learning_rate=0.02)
print(model.generate("o gato", max_new_tokens=8, temperature=0.7))
```

## O que está incluído

- **Redes totalmente ligadas** (`neural/`, `models/`): ativações `relu`,
  `leaky_relu`, `sigmoid`, `tanh`, `linear`; inicialização Xavier/He com seed.
- **Aprendizagem** (`learning/`): MSE, backpropagation, gradiente numérico
  e otimizadores **SGD, Momentum e Adam**.
- **Treino** (`training/`): mini-batches, shuffle, conjunto de validação,
  early stopping (restaura os melhores pesos), gradient clipping, callbacks.
- **Transformer treinável** (`transformer/`): embeddings, codificação
  posicional, atenção causal, feed-forward, residuais e layer norm, com
  *backpropagation* completo escrito à mão (verificado contra gradientes
  numéricos nos testes) e treino com Adam; geração com temperatura e top-k.
- **Classificação**: softmax + entropia cruzada, acurácia, matriz de
  confusão, precisão/recall/F1 por classe, CSV com rótulos de texto.
- **Dados** (`data/`): datasets validados, CSV, divisão treino/validação,
  normalização (`Scaler`: standard/minmax) e datasets de exemplo
  (XOR, AND, OR, seno, linear, blobs).
- **Avaliação** (`evaluation/`): MSE, MAE, RMSE, relatório por amostra,
  comparação de modelos; perplexidade para modelos de linguagem.
- **Checkpoints** (`checkpoints/`): JSON, escrita atómica, validação de
  ficheiros corrompidos, redes e modelos de linguagem.
- **Registry** e **AlphaCore** para gerir tudo num só sítio.

## Aceleração numérica e gráficos

Quando instalados, NumPy acelera operações matriciais e SciPy fornece um
softmax estável. Matplotlib permite guardar curvas de aprendizagem com
`--plot` (por exemplo `python main.py demo xor --plot`). O projeto não depende
de Pandas; CSV é tratado pela biblioteca padrão `csv`. Para instruções
específicas do Termux, consulta [TERMUX.md](TERMUX.md).

## Interfaces adicionais (v1.2)

- **Dashboard web local:** instala `requirements-web.txt` e executa `python webapp.py`; detalhes em [WEB.md](WEB.md).
- **Android (Kivy):** consulta [ANDROID.md](ANDROID.md) para preparar o projeto e compilar um APK com Buildozer.
- O motor principal continua sem dependências externas; Flask e Kivy são opcionais.

## Testes

```bash
python -m unittest discover -s tests -t .
```

(executar a partir da pasta do projeto.)

## Estrutura

```
main.py            ponto de entrada (menu + comandos)
cli/               menu interativo e comandos
core/              AlphaCore
alpha_math/        tensor, ativações, álgebra linear
neural/ models/    neurónio, camada, rede, fábrica
learning/          loss, backprop, gradientes, otimizadores
training/          Trainer (redes) e LanguageModelTrainer
transformer/       atenção, camada, posições, modelo de linguagem
language/          tokenizer, vocabulário, embedding
data/              Dataset, CSV, exemplos
evaluation/        métricas, avaliador, relatório
checkpoints/       guardar/carregar modelos
examples/          corpus de texto e CSV de exemplo
tests/             319 testes
```

## Limitações

É um projeto educativo: apesar da aceleração NumPy, o treino e o Transformer
ainda não são equivalentes em desempenho a PyTorch/TensorFlow. Recomenda-se
começar com modelos pequenos em telemóveis ARMv7. A atenção tem uma só cabeça
e não inclui projeções Q/K/V aprendidas.
