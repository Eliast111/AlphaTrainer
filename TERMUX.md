# AlphaTrainer no Termux (Android)

Este projeto foi ajustado para o ambiente Termux com Python 3.14 e ARMv7.
O motor funciona sem Pandas. O Pandas nao e dependencia do AlphaTrainer.

## Bibliotecas

Instalar/confirmar pelos pacotes do Termux (nao usar pip para substituir pacotes nativos):

```bash
pkg update
pkg install python python-numpy python-scipy matplotlib
```

Se algum pacote ja estiver instalado, o Termux informa que esta atualizado.
Nao instales Pandas para executar o projeto.

## Diagnostico

Na pasta `AlphaTrainer`:

```bash
python main.py doctor
python -m unittest discover -s tests -t .
```

## Usar aceleracao e graficos

O AlphaTrainer usa NumPy automaticamente nas operacoes matriciais quando a
biblioteca esta importavel. SciPy acelera o softmax. Se uma biblioteca nao
estiver disponivel, o motor usa uma implementacao Python alternativa.

```bash
python main.py demo xor --plot
python main.py train --data examples/housing.csv --layers 2,6,1 --epochs 400 --plot
```

Os graficos PNG sao guardados em `reports/`. Tambem podes indicar um caminho:

```bash
python main.py demo seno --plot reports/seno.png
```

## Exportar dados para NumPy

```python
from data.samples import load_sample
dataset = load_sample("xor")
X, y = dataset.to_numpy()
print(X.shape, y.shape)
```

## Android APK

A compilacao Kivy/Buildozer e separada do Termux. SciPy e Matplotlib nao sao
incluidos no APK por serem pesados e terem receitas Android mais exigentes.
O APK mantem o motor Python compativel; o Buildozer usa `android_app/src`.
