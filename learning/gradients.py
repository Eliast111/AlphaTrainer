"""
Alpha Learning Engine V0.1
Calculo numerico de gradientes.
"""


def numerical_gradient(function, parameters, epsilon=1e-5):
    if epsilon <= 0:
        raise ValueError("Epsilon deve ser positivo.")

    gradients = []

    for index in range(len(parameters)):
        plus = list(parameters)
        minus = list(parameters)

        plus[index] += epsilon
        minus[index] -= epsilon

        value_plus = float(function(plus))
        value_minus = float(function(minus))

        gradient = (value_plus - value_minus) / (2.0 * epsilon)
        gradients.append(gradient)

    return gradients
