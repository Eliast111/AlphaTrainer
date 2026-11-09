import json
import math
import os
import tempfile
import unittest

from checkpoints import save_model, load_model, read_metadata
from core import AlphaCore
from data.dataset import Dataset
from data.preprocessing import Scaler, ScaledModel
from data.samples import load_sample, CLASSIFICATION
from evaluation.classification import classification_report
from learning.backprop import backpropagate
from learning.loss import softmax, cross_entropy
from learning.optimizer import Adam
from models.factory import create_model
from training.trainer import Trainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLORES = os.path.join(ROOT, "examples", "flores.csv")


class TestSoftmaxAndLoss(unittest.TestCase):

    def test_softmax_sums_to_one_and_is_stable(self):
        p = softmax([1000.0, 1001.0, 999.0])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertGreater(p[1], p[0])

    def test_cross_entropy(self):
        self.assertAlmostEqual(cross_entropy([0.5, 0.5], [1, 0]), math.log(2))
        self.assertGreater(cross_entropy([0.0, 1.0], [1, 0]), 20)  # sem inf
        with self.assertRaises(ValueError):
            cross_entropy([0.5], [1, 0])

    def test_backprop_gradients_match_numerical(self):
        network = create_model([3, 4, 3], activation="tanh",
                               output_activation="linear", seed=2)
        inputs, targets = [0.3, -0.5, 0.8], [0.0, 1.0, 0.0]
        grads = backpropagate(network, inputs, targets,
                              loss="cross_entropy")["gradients"]

        def loss():
            return cross_entropy(softmax(network.forward(inputs)), targets)

        eps = 1e-6
        for layer, layer_grads in zip(network.layers, grads):
            for neuron, grad in zip(layer.neurons, layer_grads):
                for i in range(len(neuron.weights)):
                    old = neuron.weights[i]
                    neuron.weights[i] = old + eps
                    plus = loss()
                    neuron.weights[i] = old - eps
                    minus = loss()
                    neuron.weights[i] = old
                    self.assertAlmostEqual(
                        (plus - minus) / (2 * eps), grad["weights"][i], places=6
                    )

    def test_cross_entropy_requires_linear_output(self):
        network = create_model([2, 2], activation="tanh")
        with self.assertRaises(ValueError):
            backpropagate(network, [1.0, 2.0], [1.0, 0.0], loss="cross_entropy")

    def test_invalid_loss(self):
        network = create_model([2, 2], output_activation="linear")
        with self.assertRaises(ValueError):
            backpropagate(network, [1.0, 2.0], [1.0, 0.0], loss="hinge")


class TestClassificationTraining(unittest.TestCase):

    def test_blobs_reach_high_accuracy(self):
        data = load_sample("blobs")
        model = create_model([2, 8, 3], activation="tanh",
                             output_activation="linear")
        result = Trainer(Adam(0.02)).fit(
            model, data, epochs=60, batch_size=8, loss="cross_entropy"
        )
        self.assertGreater(result["final_accuracy"], 0.95)
        self.assertEqual(len(result["accuracy_history"]), 60)
        report = classification_report(model, data, CLASSIFICATION["blobs"])
        self.assertGreater(report["accuracy"], 0.95)

    def test_validation_accuracy(self):
        train, val = load_sample("blobs").split(0.25, seed=1)
        model = create_model([2, 6, 3], activation="tanh",
                             output_activation="linear")
        result = Trainer(Adam(0.02)).fit(
            model, train, epochs=10, validation_data=val,
            loss="cross_entropy"
        )
        self.assertEqual(len(result["val_accuracy_history"]), 10)

    def test_invalid_loss_name(self):
        model = create_model([2, 2], output_activation="linear")
        with self.assertRaises(ValueError):
            Trainer(Adam(0.1)).fit(model, load_sample("blobs"), loss="x")

    def test_negative_targets_rejected(self):
        model = create_model([1, 2], output_activation="linear")
        with self.assertRaises(ValueError):
            Trainer(Adam(0.1)).fit(model, [([1.0], [-1.0, 2.0])],
                                   loss="cross_entropy", epochs=1)

    def test_predict_class_and_proba(self):
        model = create_model([2, 3], output_activation="linear")
        probabilities = model.predict_proba([0.1, 0.2])
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(
            model.predict_class([0.1, 0.2]),
            probabilities.index(max(probabilities))
        )

    def test_core_helpers(self):
        alpha = AlphaCore()
        alpha.create_model("m", [2, 6, 3], activation="tanh",
                           output_activation="linear")
        data = load_sample("blobs")
        alpha.train_model("m", data, Adam(0.02), epochs=40, batch_size=8,
                          loss="cross_entropy")
        report = alpha.evaluate_classification("m", data, ["A", "B", "C"])
        self.assertGreater(report["accuracy"], 0.9)
        self.assertIn(alpha.predict_class("m", [-1.0, -1.0]), (0, 1, 2))


class TestClassificationReport(unittest.TestCase):

    class Fixed:
        def __init__(self, outputs):
            self.outputs = outputs

        def forward(self, inputs):
            return self.outputs[inputs[0]]

    def test_confusion_matrix_and_metrics(self):
        data = Dataset([
            ([0.0], [1.0, 0.0]), ([1.0], [1.0, 0.0]),
            ([2.0], [0.0, 1.0]), ([3.0], [0.0, 1.0]),
        ])
        model = self.Fixed({0.0: [1, 0], 1.0: [0, 1], 2.0: [0, 1], 3.0: [0, 1]})
        report = classification_report(model, data, ["x", "y"])
        self.assertEqual(report["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["per_class"][0]["precision"], 1.0)
        self.assertAlmostEqual(report["per_class"][0]["recall"], 0.5)

    def test_validation(self):
        data = Dataset([([0.0], [1.0, 0.0])])
        with self.assertRaises(ValueError):
            classification_report(self.Fixed({}), data, ["only_one"])
        with self.assertRaises(ValueError):
            classification_report(self.Fixed({0.0: [1]}), data)


class TestCsvLabels(unittest.TestCase):

    def test_flores(self):
        data, classes = Dataset.from_csv_labels(FLORES)
        self.assertEqual(classes, ["grande", "media", "pequena"])
        self.assertEqual(len(data), 18)
        self.assertEqual(data.target_size, 3)
        for _, target in data:
            self.assertEqual(sum(target), 1.0)

    def test_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            def write(text):
                path = os.path.join(folder, "d.csv")
                with open(path, "w") as f:
                    f.write(text)
                return path

            for bad in ("a,b\n1,x\n2,x\n",          # 1 classe
                        "a,b\nabc,x\n1,y\n",        # entrada nao numerica
                        "a,b\n1,\n2,y\n",           # rotulo vazio
                        "a,b\n",                    # vazio
                        "a\n1\n"):                  # so uma coluna
                with self.assertRaises(ValueError):
                    Dataset.from_csv_labels(write(bad))

    def test_numeric_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "d.csv")
            with open(path, "w") as f:
                f.write("x,label\n0.1,1\n0.9,0\n0.5,1\n")
            _, classes = Dataset.from_csv_labels(path)
            self.assertEqual(classes, ["0", "1"])


class TestScaler(unittest.TestCase):

    def data(self):
        return Dataset([
            ([1.0, 100.0], [10.0]), ([2.0, 200.0], [20.0]),
            ([3.0, 300.0], [30.0]),
        ])

    def test_standard(self):
        scaler = Scaler("standard").fit(self.data())
        out = scaler.transform(self.data())
        for column in range(2):
            values = [s[0][column] for s in out]
            self.assertAlmostEqual(sum(values) / 3, 0.0)
            self.assertAlmostEqual(sum(v * v for v in values) / 3, 1.0)
        self.assertEqual([s[1] for s in out], [[10.0], [20.0], [30.0]])

    def test_minmax_with_targets_and_inverse(self):
        scaler = Scaler("minmax", scale_targets=True).fit(self.data())
        out = scaler.transform(self.data())
        self.assertEqual([s[0][1] for s in out], [0.0, 0.5, 1.0])
        self.assertEqual([s[1][0] for s in out], [0.0, 0.5, 1.0])
        self.assertEqual(scaler.inverse_targets([0.5]), [20.0])

    def test_constant_column_does_not_divide_by_zero(self):
        data = Dataset([([1.0, 5.0], [1.0]), ([2.0, 5.0], [2.0])])
        out = Scaler("standard").fit(data).transform(data)
        self.assertEqual([s[0][1] for s in out], [0.0, 0.0])

    def test_serialization_roundtrip(self):
        scaler = Scaler("standard", scale_targets=True).fit(self.data())
        again = Scaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
        self.assertEqual(scaler.transform_inputs([2.0, 250.0]),
                         again.transform_inputs([2.0, 250.0]))
        self.assertEqual(scaler.inverse_targets([0.3]),
                         again.inverse_targets([0.3]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            Scaler("log")
        with self.assertRaises(ValueError):
            Scaler().transform_inputs([1.0])
        scaler = Scaler().fit(self.data())
        with self.assertRaises(ValueError):
            scaler.transform_inputs([1.0])
        with self.assertRaises(TypeError):
            Scaler().fit([1, 2])
        with self.assertRaises(ValueError):
            Scaler.from_dict({"input_stats": [[0.0, 0.0]]})
        with self.assertRaises(ValueError):
            Scaler.from_dict("x")

    def test_scaled_model_uses_original_units(self):
        data = self.data()
        scaler = Scaler("minmax", scale_targets=True).fit(data)
        model = create_model([2, 1], activation="linear")
        Trainer(Adam(0.1)).fit(model, scaler.transform(data), epochs=300)
        wrapped = ScaledModel(model, scaler)
        self.assertAlmostEqual(wrapped.forward([2.0, 200.0])[0], 20.0, delta=1.0)

    def test_scaling_helps_badly_scaled_data(self):
        # Entradas em escalas muito diferentes: sem normalizar o treino falha.
        data = Dataset([
            ([x / 10.0, x * 1000.0], [x / 10.0]) for x in range(1, 9)
        ])
        scaler = Scaler("standard").fit(data)
        scaled = Trainer(Adam(0.05)).fit(
            create_model([2, 1], activation="linear", seed=1),
            scaler.transform(data), epochs=200
        )["final_loss"]
        self.assertLess(scaled, 0.01)


class TestMetadata(unittest.TestCase):

    def test_metadata_roundtrip_and_backward_compat(self):
        model = create_model([2, 2], output_activation="linear")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.json")
            save_model(model, path, metadata={"task": "classification",
                                              "classes": ["a", "b"]})
            self.assertEqual(read_metadata(path)["classes"], ["a", "b"])
            load_model(path)  # metadados nao atrapalham o carregamento

            save_model(model, path)
            self.assertEqual(read_metadata(path), {})

            with self.assertRaises(TypeError):
                save_model(model, path, metadata=[1])


if __name__ == "__main__":
    unittest.main()
