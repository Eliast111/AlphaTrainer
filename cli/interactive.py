"""
Alpha CLI - menu interativo.

As funcoes de entrada/saida sao injetaveis para permitir testes.
"""

import os

from core import AlphaCore
from data.samples import available_samples
from neural.network import Network
from transformer.language_model import LanguageModel
from checkpoints import read_metadata
from data.preprocessing import Scaler
from cli.common import (
    DEFAULT_MODELS_DIR, parse_sizes, parse_floats, load_task_dataset,
    train_network, print_evaluation, read_text_corpus, make_progress,
    model_metadata
)

VERSION = "1.1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(PROJECT_ROOT, "examples", "corpus_pt.txt")


class Quit(Exception):
    """Pedido para terminar o programa."""


class Menu:
    def __init__(self, alpha=None, input_fn=input, output_fn=print):
        self.alpha = alpha or AlphaCore()
        self.input = input_fn
        self.out = output_fn
        self.datasets = {}
        self.dataset_info = {}   # nome -> {"task", "classes"}
        self.model_info = {}     # nome -> {"task", "classes", "scaler"}

    # ------------------------------------------------------------------
    # Entrada

    def ask(self, prompt, default=None, cast=str):
        suffix = f" [{default}]" if default not in (None, "") else ""

        try:
            raw = self.input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            raise Quit()

        if not raw:
            if default is None:
                return ""
            return cast(default)

        return cast(raw)

    def choose_model(self, kind=None, prompt="Nome do modelo"):
        names = self.alpha.list_models()

        if kind is not None:
            names = [
                n for n in names if isinstance(self.alpha.get_model(n), kind)
            ]

        if not names:
            raise ValueError("Nao existem modelos adequados. Crie ou carregue um.")

        self.out("Disponiveis: " + ", ".join(names))
        default = names[0] if len(names) == 1 else None
        name = self.ask(prompt, default)

        if name not in names:
            raise ValueError(f"Modelo {name!r} nao encontrado.")

        return name

    def choose_dataset(self):
        if not self.datasets:
            raise ValueError(
                "Ainda nao ha datasets carregados (opcao 'Carregar dataset')."
            )

        names = list(self.datasets)
        self.out("Datasets: " + ", ".join(names))
        name = self.ask("Dataset", names[0] if len(names) == 1 else None)

        if name not in self.datasets:
            raise ValueError(f"Dataset {name!r} nao encontrado.")

        return name, self.datasets[name]

    # ------------------------------------------------------------------
    # Apresentacao

    def banner(self):
        self.out("\n" + "=" * 40)
        self.out("          ALPHATECHNOLOGIES")
        self.out("             AlphaTrainer")
        self.out(f"                 V{VERSION}")
        self.out("=" * 40)

    MENU = [
        ("1", "Criar rede neuronal"),
        ("2", "Listar modelos"),
        ("3", "Remover modelo"),
        ("4", "Carregar dataset (exemplo ou CSV)"),
        ("5", "Treinar rede"),
        ("6", "Avaliar rede (relatorio)"),
        ("7", "Fazer previsao"),
        ("8", "Guardar modelo"),
        ("9", "Carregar modelo guardado"),
        ("10", "Criar e treinar modelo de linguagem"),
        ("11", "Gerar texto"),
        ("12", "Estado do sistema"),
        ("0", "Sair"),
    ]

    def show_menu(self):
        self.out("\nMENU PRINCIPAL")

        for key, label in self.MENU:
            self.out(f"{key}. {label}")

    # ------------------------------------------------------------------
    # Acoes

    def create_model(self):
        self.out("\nCRIAR REDE NEURONAL")
        name = self.ask("Nome do modelo")
        sizes = self.ask(
            "Tamanhos das camadas (entradas,...,saidas; ex.: 2,4,1)",
            cast=parse_sizes
        )
        activation = self.ask(
            "Ativacao oculta (relu, leaky_relu, sigmoid, tanh, linear)",
            "tanh", str.lower
        )
        output_activation = self.ask(
            "Ativacao de saida (linear para regressao, sigmoid para 0/1)",
            "linear", str.lower
        )

        self.alpha.create_model(
            name, sizes, activation=activation,
            output_activation=output_activation
        )
        info = self.alpha.describe_model(name)
        self.out(
            f"\nModelo criado! Arquitetura {info['layers']}, "
            f"{info['parameters']} parametros."
        )

    def list_models(self):
        self.out("\nMODELOS REGISTADOS")
        names = self.alpha.list_models()

        if not names:
            self.out("Ainda nao existem modelos registados.")
            return

        for index, name in enumerate(names, start=1):
            info = self.alpha.describe_model(name)
            extra = (
                f"camadas {info['layers']}"
                if "layers" in info
                else f"vocabulario {info.get('vocabulary', '?')}"
            )
            self.out(
                f"{index}. {name} - {info['type']}, {extra}, "
                f"{info.get('parameters', '?')} parametros"
            )

        self.out(f"Total: {self.alpha.model_count()}")

    def remove_model(self):
        self.out("\nREMOVER MODELO")
        name = self.ask("Nome do modelo")
        self.alpha.remove_model(name)
        self.model_info.pop(name, None)
        self.out("Modelo removido com sucesso!")

    def load_dataset(self):
        self.out("\nCARREGAR DATASET")
        self.out("Exemplos: " + ", ".join(
            f"{k} ({v})" for k, v in available_samples().items()
        ))
        spec = self.ask("Nome do exemplo ou caminho do CSV", "xor")
        task = "auto"
        targets = 1

        if os.path.isfile(spec):
            task = self.ask(
                "Tarefa (regression = ultimas colunas sao alvos numericos; "
                "classification = ultima coluna e o rotulo)",
                "regression", str.lower
            )

            if task == "regression":
                targets = self.ask("Numero de colunas alvo (ultimas)", 1, int)

        dataset, classes, task = load_task_dataset(spec, task, targets)
        name = self.ask(
            "Guardar com o nome",
            os.path.splitext(os.path.basename(spec))[0]
        )
        self.datasets[name] = dataset
        self.dataset_info[name] = {"task": task, "classes": classes}
        self.out(
            f"Dataset {name!r} ({task}): {len(dataset)} amostras, "
            f"{dataset.input_size} entradas, {dataset.target_size} saidas."
        )

        if classes:
            self.out("Classes: " + ", ".join(classes))

    def train(self):
        self.out("\nTREINAR REDE")
        name = self.choose_model(Network)
        data_name, dataset = self.choose_dataset()
        model = self.alpha.get_model(name)

        if (
            model.input_size() != dataset.input_size
            or model.output_size() != dataset.target_size
        ):
            raise ValueError(
                f"Modelo ({model.input_size()} -> {model.output_size()}) "
                f"incompativel com o dataset "
                f"({dataset.input_size} -> {dataset.target_size})."
            )

        info = self.dataset_info.get(data_name, {"task": "regression"})
        classification = info["task"] == "classification"

        if classification and any(
            n.activation_name != "linear" for n in model.layers[-1].neurons
        ):
            raise ValueError(
                "Para classificar, a ativacao de saida da rede deve ser "
                "linear (o softmax e aplicado automaticamente)."
            )

        scale = self.ask("Normalizacao das entradas (none, standard, minmax)",
                         "none", str.lower)
        optimizer = self.ask("Otimizador (sgd, momentum, adam)", "adam", str.lower)
        lr = self.ask("Learning rate (vazio = padrao do otimizador)", "", str)
        epochs = self.ask("Epocas", 500, int)
        batch = self.ask("Tamanho do batch", 1, int)

        result = train_network(
            self.alpha, name, dataset, optimizer=optimizer,
            learning_rate=float(lr) if lr else None,
            epochs=epochs, batch_size=batch, output=self.out,
            loss="cross_entropy" if classification else "mse", scale=scale
        )
        self.model_info[name] = {
            "task": info["task"], "classes": info.get("classes"),
            "scaler": result["scaler"],
        }
        self.out(f"Perda final: {result['final_loss']:.6f}")

        if classification:
            self.out(f"Acuracia no treino: {result['final_accuracy'] * 100:.1f}%")

    def evaluate(self):
        self.out("\nAVALIAR REDE")
        name = self.choose_model(Network)
        _, dataset = self.choose_dataset()
        info = self.model_info.get(name, {})
        print_evaluation(
            self.alpha, name, dataset, output=self.out,
            scaler=info.get("scaler"), classes=info.get("classes"),
            task=info.get("task", "regression")
        )

    def predict(self):
        self.out("\nFAZER PREVISAO")
        name = self.choose_model(Network)
        size = self.alpha.get_model(name).input_size()
        values = self.ask(f"Entradas ({size} valores separados por virgula)",
                          cast=parse_floats)
        info = self.model_info.get(name, {})
        scaler = info.get("scaler")
        model = self.alpha.get_model(name)

        if scaler is not None:
            values = scaler.transform_inputs(values)

        if info.get("task") == "classification":
            classes = info.get("classes") or [
                str(i) for i in range(model.output_size())
            ]
            probabilities = model.predict_proba(values)
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            self.out(f"Classe prevista: {classes[best]}")

            for label, p in zip(classes, probabilities):
                self.out(f"  {label}: {p * 100:.1f}%")

            return

        result = model.forward(values)

        if scaler is not None:
            result = scaler.inverse_targets(result)

        self.out("Saida: " + ", ".join(f"{v:.6f}" for v in result))

    def save(self):
        self.out("\nGUARDAR MODELO")
        name = self.choose_model()
        path = self.ask(
            "Ficheiro", os.path.join(DEFAULT_MODELS_DIR, f"{name}.json")
        )
        info = self.model_info.get(name)
        metadata = (
            model_metadata(info["task"], info["classes"], info["scaler"])
            if info else None
        )
        self.alpha.save_model(name, path, metadata=metadata)
        self.out(f"Modelo guardado em {path}")

    def load(self):
        self.out("\nCARREGAR MODELO")
        path = self.ask("Ficheiro (.json)")
        name = self.ask(
            "Registar com o nome",
            os.path.splitext(os.path.basename(path))[0]
        )
        self.alpha.load_model(name, path)

        try:
            metadata = read_metadata(path)
        except (ValueError, KeyError):
            metadata = {}

        if metadata.get("task"):
            self.model_info[name] = {
                "task": metadata["task"],
                "classes": metadata.get("classes"),
                "scaler": (
                    Scaler.from_dict(metadata["scaler"])
                    if metadata.get("scaler") else None
                ),
            }

        self.out(f"Modelo {name!r} carregado.")

    def language_model(self):
        self.out("\nMODELO DE LINGUAGEM")
        name = self.ask("Nome do modelo", "lm")
        path = self.ask("Ficheiro de texto (uma frase por linha)",
                        DEFAULT_CORPUS)
        texts = read_text_corpus(path)
        epochs = self.ask("Epocas", 60, int)
        lr = self.ask("Learning rate", 0.02, float)

        self.alpha.create_language_model(
            name, texts, embedding_dim=16, hidden_dim=32,
            num_layers=1, max_length=32
        )
        info = self.alpha.describe_model(name)
        self.out(
            f"Vocabulario: {info['vocabulary']} tokens, "
            f"{info['parameters']} parametros. A treinar..."
        )

        progress = make_progress(epochs, self.out)

        result = self.alpha.train_language_model(
            name, texts, epochs=epochs, learning_rate=lr, batch_size=4,
            callback=lambda e, l: progress(e, l)
        )
        self.out(
            f"Perda final {result['final_loss']:.4f} "
            f"(perplexidade {result['perplexity']:.2f})"
        )

    def generate(self):
        self.out("\nGERAR TEXTO")
        name = self.choose_model(LanguageModel)
        prompt = self.ask("Inicio do texto (pode ficar vazio)", "")
        tokens = self.ask("Maximo de palavras", 12, int)
        temperature = self.ask("Temperatura (0 = deterministico)", 0.7, float)
        self.out("\n" + self.alpha.generate(
            name, prompt, max_new_tokens=tokens, temperature=temperature
        ))

    def status(self):
        self.out("\nESTADO DO ALPHATRAINER")
        self.out(f"Versao: {VERSION}")
        self.out("Nucleo: AlphaCore")
        self.out("Registry: ativo")
        self.out(f"Modelos registados: {self.alpha.model_count()}")
        self.out(f"Datasets carregados: {len(self.datasets)}")
        self.out("Estado: operacional")

    # ------------------------------------------------------------------

    def run(self):
        actions = {
            "1": self.create_model, "2": self.list_models,
            "3": self.remove_model, "4": self.load_dataset,
            "5": self.train, "6": self.evaluate, "7": self.predict,
            "8": self.save, "9": self.load, "10": self.language_model,
            "11": self.generate, "12": self.status,
        }

        self.banner()

        while True:
            self.show_menu()

            try:
                option = self.ask("\nEscolhe uma opcao")

                if option == "0":
                    self.out("\nAlphaTrainer encerrado.")
                    return

                action = actions.get(option)

                if action is None:
                    self.out("\nOpcao invalida. Tenta novamente.")
                    continue

                action()

            except Quit:
                self.out("\nAlphaTrainer encerrado.")
                return

            except KeyboardInterrupt:
                self.out("\nOperacao cancelada.")

            except (ValueError, TypeError, KeyError, OSError) as error:
                message = error.args[0] if isinstance(error, KeyError) and error.args else error
                self.out(f"\nErro: {message}")
