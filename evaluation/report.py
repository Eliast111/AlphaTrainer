"""
Alpha Evaluation Engine V0.3
Geracao de relatorios de desempenho.
"""

from evaluation.evaluator import evaluate_model


def generate_report(network, dataset):
    result = evaluate_model(network, dataset)

    target_size = dataset.target_size
    sample_reports = []

    for index in range(result["samples"]):
        start = index * target_size
        end = start + target_size

        predictions = result["predictions"][start:end]
        targets = result["targets"][start:end]

        errors = [
            abs(prediction - target)
            for prediction, target in zip(predictions, targets)
        ]

        sample_mae = sum(errors) / len(errors)

        sample_reports.append({
            "sample": index + 1,
            "predictions": predictions,
            "targets": targets,
            "absolute_errors": errors,
            "mae": sample_mae,
        })

    worst_sample = max(
        sample_reports,
        key=lambda item: item["mae"]
    )

    return {
        "summary": {
            "samples": result["samples"],
            "mse": result["metrics"]["mse"],
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
        },
        "sample_reports": sample_reports,
        "worst_sample": worst_sample["sample"],
        "worst_sample_mae": worst_sample["mae"],
    }
