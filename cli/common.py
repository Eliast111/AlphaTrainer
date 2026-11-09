"""
Alpha CLI - funcoes partilhadas pelo menu interativo e pelos comandos.
"""

import os

from data.dataset import Dataset
from data.preprocessing import Scaler, ScaledModel
from data.samples import SAMPLES, CLASSIFICATION, load_sample
from evaluation.classification import classification_report
from evaluation.report import generate_report
from learning.optimizer import create_optimizer

DEFAULT_MODELS_DIR = "saved_models"
TASKS = ("regression", "classification")


def parse_sizes(text):
    try:
        sizes = [int(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError:
        raise ValueError("Use inteiros separados por virgulas, ex.: 2,4,1")

    if len(sizes) < 2:
        raise ValueError("Indique pelo menos entradas e saidas, ex.: 2,4,1")

    return sizes


def parse_floats(text):
    try:
        values = [float(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError:
        raise ValueError("Use numeros separados por virgulas, ex.: 1,0")

    if not values:
        raise ValueError("Indique pelo menos um numero.")

    return values


def load_dataset(spec, target_columns=1, delimiter=","):
    """spec: nome de um dataset de exemplo ou caminho de um CSV."""
    key = str(spec).strip().lower()

    if key in SAMPLES:
        return load_sample(key)

    if not os.path.isfile(spec):
        raise ValueError(
            f"Dataset {spec!r} nao encontrado. Exemplos: "
            + ", ".join(SAMPLES)
            + " ou o caminho de um ficheiro CSV."
        )

    return Dataset.from_csv(
        spec, target_columns=target_columns, delimiter=delimiter
    )


def load_task_dataset(spec, task="auto", target_columns=1, delimiter=","):
    """Carrega um dataset e devolve (dataset, classes, task).

    task: "auto" | "regression" | "classification".
    Em classificacao, um CSV tem o rotulo (texto ou numero) na ultima coluna.
    """
    if task not in ("auto",) + TASKS:
        raise ValueError("task deve ser auto, regression ou classification.")

    key = str(spec).strip().lower()

    if key in SAMPLES:
        classes = CLASSIFICATION.get(key)
        actual = "classification" if classes else "regression"

        if task not in ("auto", actual):
            raise ValueError(
                f"O exemplo {key!r} e de {actual}, nao de {task}."
            )

        return load_sample(key), classes, actual

    if task == "classification":
        if not os.path.isfile(spec):
            raise ValueError(f"Dataset {spec!r} nao encontrado.")

        dataset, classes = Dataset.from_csv_labels(spec, delimiter=delimiter)
        return dataset, classes, "classification"

    return load_dataset(spec, target_columns, delimiter), None, "regression"


def make_progress(epochs, output=print, steps=10):
    """Callback que imprime o progresso ~`steps` vezes."""
    every = max(1, epochs // steps)

    def callback(epoch, train_loss, val_loss=None):
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            line = f"  epoca {epoch:>6}/{epochs}  perda treino {train_loss:.6f}"

            if val_loss is not None:
                line += f"  validacao {val_loss:.6f}"

            output(line)

        return False

    return callback


def train_network(
    alpha,
    name,
    dataset,
    optimizer="adam",
    learning_rate=None,
    epochs=500,
    batch_size=1,
    validation=0.0,
    patience=None,
    seed=0,
    output=print,
    loss="mse",
    scale="none",
    scale_targets=False
):
    """Treina um modelo registado e devolve o resultado do trainer.

    Com scale != "none" as entradas sao normalizadas (ajuste so nos dados de
    treino); o Scaler fica em result["scaler"].
    """
    train_set = dataset
    validation_set = None

    if validation and validation > 0:
        train_set, validation_set = dataset.split(validation, seed=seed)

    scaler = None

    if scale and scale != "none":
        scaler = Scaler(scale, scale_targets=scale_targets).fit(train_set)
        train_set = scaler.transform(train_set)

        if validation_set is not None:
            validation_set = scaler.transform(validation_set)

    opt = create_optimizer(optimizer, learning_rate)

    result = alpha.train_model(
        name,
        train_set,
        opt,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_set,
        patience=patience,
        seed=seed,
        callback=make_progress(epochs, output),
        loss=loss
    )

    result["scaler"] = scaler
    return result


def print_evaluation(
    alpha, name, dataset, output=print, max_rows=12,
    scaler=None, classes=None, task="regression"
):
    """Avalia no dataset original (com o Scaler, se existir)."""
    model = alpha.get_model(name)

    if scaler is not None:
        model = ScaledModel(model, scaler)

    if task == "classification":
        report = classification_report(model, dataset, classes)
        output(
            f"Amostras: {report['samples']}  "
            f"Acuracia: {report['accuracy'] * 100:.1f}%"
        )
        output("Matriz de confusao (linhas = real, colunas = previsto):")
        width = max(len(c) for c in report["classes"]) + 2
        output(" " * width + "".join(c.rjust(width) for c in report["classes"]))

        for label, row in zip(report["classes"], report["confusion_matrix"]):
            output(label.ljust(width) + "".join(str(v).rjust(width) for v in row))

        for item in report["per_class"]:
            output(
                f"  {item['class']}: precisao {item['precision']:.2f}  "
                f"recall {item['recall']:.2f}  f1 {item['f1']:.2f}  "
                f"(n={item['support']})"
            )

        return report

    report = generate_report(model, dataset)
    summary = report["summary"]

    output(
        f"Amostras: {summary['samples']}  MSE: {summary['mse']:.6f}  "
        f"MAE: {summary['mae']:.6f}  RMSE: {summary['rmse']:.6f}"
    )

    for item in report["sample_reports"][:max_rows]:
        predictions = ", ".join(f"{v:.4f}" for v in item["predictions"])
        targets = ", ".join(f"{v:.4f}" for v in item["targets"])
        output(
            f"  #{item['sample']:<3} previsto [{predictions}]  "
            f"esperado [{targets}]"
        )

    if len(report["sample_reports"]) > max_rows:
        output(f"  ... (+{len(report['sample_reports']) - max_rows} amostras)")

    return report


def model_metadata(task, classes, scaler):
    return {
        "task": task,
        "classes": classes,
        "scaler": scaler.to_dict() if scaler is not None else None,
    }


def read_text_corpus(path):
    if not os.path.isfile(path):
        raise ValueError(f"Ficheiro {path!r} nao encontrado.")

    with open(path, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError("O ficheiro de texto esta vazio.")

    return lines
