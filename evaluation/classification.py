"""
Alpha Evaluation Engine - Classificacao
Acuracia, matriz de confusao e metricas por classe.
"""


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def classification_report(network, dataset, classes=None):
    if not callable(getattr(network, "forward", None)):
        raise TypeError("O modelo deve possuir o metodo forward.")

    if not dataset:
        raise ValueError("O dataset nao pode estar vazio.")

    size = dataset.target_size

    if size < 2:
        raise ValueError("Classificacao exige pelo menos 2 classes.")

    if classes is None:
        classes = [str(i) for i in range(size)]

    if len(classes) != size:
        raise ValueError("Numero de classes incompativel com os alvos.")

    matrix = [[0] * size for _ in range(size)]

    for inputs, targets in dataset:
        output = network.forward(inputs)

        if len(output) != size:
            raise ValueError("O tamanho da previsao nao corresponde aos alvos.")

        matrix[_argmax(targets)][_argmax(output)] += 1

    total = len(dataset)
    correct = sum(matrix[i][i] for i in range(size))
    per_class = []

    for i, name in enumerate(classes):
        true_positive = matrix[i][i]
        predicted = sum(matrix[r][i] for r in range(size))
        actual = sum(matrix[i])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class.append({
            "class": name, "precision": precision,
            "recall": recall, "f1": f1, "support": actual,
        })

    return {
        "samples": total,
        "accuracy": correct / total,
        "classes": list(classes),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }
