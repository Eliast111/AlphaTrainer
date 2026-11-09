"""
Alpha CLI - comandos nao interativos (python main.py <comando>).
"""

import argparse
import sys

from core import AlphaCore
from data.samples import SAMPLES, available_samples
from checkpoints import load_any
from neural.network import Network
from transformer.language_model import LanguageModel
from checkpoints import read_metadata
from data.preprocessing import Scaler, ScaledModel
from cli.common import (
    parse_sizes, parse_floats, load_task_dataset, train_network,
    print_evaluation, read_text_corpus, make_progress, model_metadata
)

SAMPLES_NAMES = set(SAMPLES)

DEMO_SETTINGS = {
    # nome: (camadas, ativacao, ativacao saida, otimizador, lr, epocas, batch)
    "xor": ([2, 4, 1], "tanh", "sigmoid", "adam", 0.05, 600, 1),
    "and": ([2, 3, 1], "tanh", "sigmoid", "adam", 0.05, 300, 1),
    "or": ([2, 3, 1], "tanh", "sigmoid", "adam", 0.05, 300, 1),
    "seno": ([1, 8, 1], "tanh", "linear", "adam", 0.01, 600, 4),
    "linear": ([1, 1], "linear", "linear", "adam", 0.05, 400, 1),
    "blobs": ([2, 8, 3], "tanh", "linear", "adam", 0.02, 120, 8),
}


def cmd_demo(args, output=print):
    alpha = AlphaCore()
    layers, act, out_act, opt, lr, epochs, batch = DEMO_SETTINGS[args.name]
    dataset, classes, task = load_task_dataset(args.name)
    loss = "cross_entropy" if task == "classification" else "mse"

    output(f"Demo '{args.name}': {available_samples()[args.name]}")
    output(f"Rede {layers}, ativacao {act}/{out_act}, {opt} lr={lr}")

    alpha.create_model(
        "demo", layers, activation=act, output_activation=out_act, seed=args.seed
    )

    result = train_network(
        alpha, "demo", dataset, optimizer=opt, learning_rate=lr,
        epochs=epochs, batch_size=batch, seed=args.seed, output=output,
        loss=loss
    )

    output(f"Perda final: {result['final_loss']:.6f}")
    output("Resultado:")
    print_evaluation(alpha, "demo", dataset, output=output,
                     classes=classes, task=task)

    if args.save:
        alpha.save_model("demo", args.save,
                         metadata=model_metadata(task, classes, None))
        output(f"Modelo guardado em {args.save}")

    if args.plot:
        from visualization.plots import save_loss_plot
        path = save_loss_plot(result["history"], args.plot, result.get("val_history"),
                              title=f"AlphaTrainer - demo {args.name}")
        output(f"Grafico guardado em {path}")
    return 0


def cmd_train(args, output=print):
    alpha = AlphaCore()
    dataset, classes, task = load_task_dataset(
        args.data, args.task, args.targets
    )
    layers = parse_sizes(args.layers)

    if layers[0] != dataset.input_size:
        raise ValueError(
            f"A primeira camada ({layers[0]}) deve ter o tamanho das "
            f"entradas do dataset ({dataset.input_size})."
        )

    if layers[-1] != dataset.target_size:
        what = (
            f"numero de classes ({dataset.target_size}: "
            + ", ".join(classes) + ")"
            if classes else f"tamanho dos alvos ({dataset.target_size})"
        )
        raise ValueError(f"A ultima camada ({layers[-1]}) deve ter o {what}.")

    classification = task == "classification"

    # Em classificacao a saida e linear (logits + softmax).
    output_activation = "linear" if classification else args.output_activation

    scale = args.scale

    if scale == "auto":
        scale = "none" if str(args.data).lower() in SAMPLES_NAMES else "standard"

    alpha.create_model(
        "modelo", layers, activation=args.activation,
        output_activation=output_activation, seed=args.seed
    )

    output(
        f"{len(dataset)} amostras, tarefa: "
        f"{'classificacao' if classification else 'regressao'}, "
        f"rede {layers}, {args.optimizer}, normalizacao: {scale}"
    )

    result = train_network(
        alpha, "modelo", dataset, optimizer=args.optimizer,
        learning_rate=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, validation=args.validation,
        patience=args.patience, seed=args.seed, output=output,
        loss="cross_entropy" if classification else "mse",
        scale=scale,
        scale_targets=args.scale_targets and not classification
    )

    scaler = result["scaler"]

    output(
        f"Treino terminou em {result['epochs_run']} epocas, "
        f"perda final {result['final_loss']:.6f}"
    )
    print_evaluation(alpha, "modelo", dataset, output=output,
                     scaler=scaler, classes=classes, task=task)

    if args.save:
        alpha.save_model("modelo", args.save,
                         metadata=model_metadata(task, classes, scaler))
        output(f"Modelo guardado em {args.save}")

    if args.plot:
        from visualization.plots import save_loss_plot
        path = save_loss_plot(result["history"], args.plot, result.get("val_history"),
                              title="AlphaTrainer - treino")
        output(f"Grafico guardado em {path}")
    return 0


def cmd_predict(args, output=print):
    model = load_any(args.model)

    if not isinstance(model, Network):
        raise ValueError("Este checkpoint e um modelo de linguagem; use 'generate'.")

    metadata = read_metadata(args.model)
    values = parse_floats(args.input)

    scaler = (
        Scaler.from_dict(metadata["scaler"]) if metadata.get("scaler") else None
    )

    if scaler is not None:
        values = scaler.transform_inputs(values)

    if metadata.get("task") == "classification":
        classes = metadata.get("classes") or [
            str(i) for i in range(model.output_size())
        ]
        probabilities = model.predict_proba(values)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)

        output(f"Classe prevista: {classes[best]}")

        for label, p in sorted(zip(classes, probabilities), key=lambda x: -x[1]):
            output(f"  {label}: {p * 100:.1f}%")

        return 0

    result = model.forward(values)

    if scaler is not None:
        result = scaler.inverse_targets(result)

    output("Saida: " + ", ".join(f"{v:.6f}" for v in result))
    return 0


def cmd_lm_train(args, output=print):
    texts = read_text_corpus(args.text)

    model = LanguageModel.from_texts(
        texts,
        embedding_dim=args.dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        max_length=args.max_length,
        seed=args.seed
    )

    output(
        f"{len(texts)} frases, vocabulario {model.vocabulary.size}, "
        f"{model.parameter_count()} parametros"
    )

    result = model.fit(
        texts,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        callback=lambda e, l: make_progress(args.epochs, output)(e, l)
    )

    output(
        f"Perda final {result['final_loss']:.4f} "
        f"(perplexidade {result['perplexity']:.2f})"
    )

    if args.prompt is not None:
        output("Exemplo: " + model.generate(
            args.prompt, max_new_tokens=args.tokens,
            temperature=args.temperature, seed=args.seed
        ))

    if args.save:
        from checkpoints import save_any
        save_any(model, args.save)
        output(f"Modelo guardado em {args.save}")

    if args.plot:
        from visualization.plots import save_loss_plot
        path = save_loss_plot(result["history"], args.plot,
                              title="AlphaTrainer - treino Transformer")
        output(f"Grafico guardado em {path}")
    return 0


def cmd_generate(args, output=print):
    model = load_any(args.model)

    if not isinstance(model, LanguageModel):
        raise ValueError("Este checkpoint nao e um modelo de linguagem.")

    output(model.generate(
        args.prompt, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed
    ))
    return 0


def cmd_info(args, output=print):
    alpha = AlphaCore()
    alpha.load_model("modelo", args.model)

    for key, value in alpha.describe_model("modelo").items():
        output(f"{key}: {value}")

    metadata = read_metadata(args.model)

    if metadata.get("task"):
        output(f"tarefa: {metadata['task']}")

    if metadata.get("classes"):
        output("classes: " + ", ".join(metadata["classes"]))

    if metadata.get("scaler"):
        output(f"normalizacao: {metadata['scaler']['method']}")

    return 0


def cmd_doctor(args, output=print):
    from alpha_math.backend import backend_info
    output("AlphaTrainer - diagnostico do ambiente")
    for name, version in backend_info().items():
        output(f"{name}: {version or 'nao instalado / nao importavel'}")
    output("Pandas: opcional (nao e necessario para treinar)")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="alphatrainer",
        description="AlphaTrainer - redes neuronais e Transformer com aceleracao NumPy/SciPy opcional."
    )
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="verifica bibliotecas e aceleradores")
    doctor.set_defaults(func=cmd_doctor)

    demo = sub.add_parser("demo", help="treina um exemplo pronto a usar")
    demo.add_argument("name", choices=sorted(SAMPLES), nargs="?", default="xor")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--save", help="guardar o modelo neste ficheiro")
    demo.add_argument("--plot", nargs="?", const="reports/demo-loss.png",
                      help="guardar grafico de perda (PNG)")
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train", help="treina uma rede com um dataset")
    train.add_argument("--data", required=True,
                       help="CSV ou exemplo: " + ", ".join(SAMPLES))
    train.add_argument("--targets", type=int, default=1,
                       help="numero de colunas alvo (ultimas do CSV)")
    train.add_argument("--task", default="auto",
                       choices=["auto", "regression", "classification"],
                       help="classification: ultima coluna do CSV = rotulo")
    train.add_argument("--scale", default="auto",
                       choices=["auto", "none", "standard", "minmax"],
                       help="normalizacao das entradas (auto: standard em CSV)")
    train.add_argument("--scale-targets", action="store_true",
                       help="normalizar tambem os alvos (regressao)")
    train.add_argument("--layers", required=True, help="ex.: 2,4,1")
    train.add_argument("--activation", default="tanh")
    train.add_argument("--output-activation", default="linear")
    train.add_argument("--optimizer", default="adam",
                       choices=["sgd", "momentum", "adam"])
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--validation", type=float, default=0.0,
                       help="proporcao para validacao (ex.: 0.2)")
    train.add_argument("--patience", type=int, default=None,
                       help="early stopping (epocas sem melhoria)")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save", help="guardar o modelo neste ficheiro")
    train.add_argument("--plot", nargs="?", const="reports/training-loss.png",
                       help="guardar grafico de perda (PNG)")
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="usa uma rede guardada")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True, help="ex.: 1,0")
    predict.set_defaults(func=cmd_predict)

    lm = sub.add_parser("lm-train", help="treina um modelo de linguagem")
    lm.add_argument("--text", required=True, help="ficheiro com uma frase por linha")
    lm.add_argument("--epochs", type=int, default=80)
    lm.add_argument("--dim", type=int, default=16)
    lm.add_argument("--hidden", type=int, default=32)
    lm.add_argument("--layers", type=int, default=1)
    lm.add_argument("--max-length", type=int, default=32)
    lm.add_argument("--lr", type=float, default=0.02)
    lm.add_argument("--batch-size", type=int, default=4)
    lm.add_argument("--seed", type=int, default=42)
    lm.add_argument("--prompt", default=None, help="gerar um exemplo no fim")
    lm.add_argument("--tokens", type=int, default=10)
    lm.add_argument("--temperature", type=float, default=0.7)
    lm.add_argument("--save", help="guardar o modelo neste ficheiro")
    lm.add_argument("--plot", nargs="?", const="reports/transformer-loss.png",
                    help="guardar grafico de perda (PNG)")
    lm.set_defaults(func=cmd_lm_train)

    gen = sub.add_parser("generate", help="gera texto com um modelo guardado")
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", default="")
    gen.add_argument("--tokens", type=int, default=15)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--seed", type=int, default=None)
    gen.set_defaults(func=cmd_generate)

    info = sub.add_parser("info", help="mostra informacao de um modelo guardado")
    info.add_argument("--model", required=True)
    info.set_defaults(func=cmd_info)

    return parser


def run(argv=None, output=print):
    """Executa um comando. Devolve o codigo de saida, ou None se nao houver
    comando (nesse caso deve arrancar o menu interativo)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        return None

    try:
        return args.func(args, output=output)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print("Erro:", error, file=sys.stderr)
        return 1
