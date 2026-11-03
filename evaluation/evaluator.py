"""
Alpha Evaluation Engine V0.2
Avaliacao de redes neurais com datasets.
"""

from evaluation.metrics import evaluate


def evaluate_model(network, dataset):
    if not callable(getattr(network, "forward", None)):
        raise TypeError("O modelo deve possuir o metodo forward.")

    if not dataset:
        raise ValueError("O dataset nao pode estar vazio.")

    predictions = []
    targets = []

    for inputs, expected in dataset:
        output = network.forward(inputs)

        if not isinstance(output, (list, tuple)):
            output = [output]

        if len(output) != len(expected):
            raise ValueError(
                "O tamanho da previsao nao corresponde aos alvos."
            )

        predictions.extend(output)
        targets.extend(expected)

    metrics = evaluate(predictions, targets)

    return {
        "samples": len(dataset),
        "predictions": predictions,
        "targets": targets,
        "metrics": metrics,
    }


def compare_models(models, dataset):
    if not isinstance(models, dict) or not models:
        raise ValueError("Forneca um dicionario de modelos.")

    results = {}

    for name, model in models.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Todos os modelos devem ter nomes validos.")

        results[name] = evaluate_model(model, dataset)

    ranking = sorted(
        results,
        key=lambda name: results[name]["metrics"]["mse"]
    )

    return {
        "results": results,
        "ranking": ranking,
        "best_model": ranking[0],
    }
