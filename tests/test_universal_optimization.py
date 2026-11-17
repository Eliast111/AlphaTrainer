import unittest

from core import AlphaCore
from data.dataset import Dataset

from optimization.universal import (
    UniversalOptimizationEngine,
    OptimizationError,
)


class FakeManager:
    pass


class TestUniversalOptimization(unittest.TestCase):

    def setUp(self):
        self.engine = UniversalOptimizationEngine(FakeManager())



    def test_dense_optimization_with_three_optimizers(self):
        from optimization.universal import UniversalOptimizationEngine

        alpha = AlphaCore()
        model = alpha.create_model(
            "modelo_otimizacao",
            [1, 1],
            activation="linear"
        )

        original_weights = [
            list(neuron.weights)
            for layer in model.layers
            for neuron in layer.neurons
        ]

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
            ([3.0], [6.0]),
        ])

        engine = UniversalOptimizationEngine(alpha.model_manager)

        result = engine.optimize(
            "modelo_otimizacao",
            dataset,
            configurations=[
                {"optimizer": "sgd", "learning_rate": 0.01},
                {"optimizer": "momentum", "learning_rate": 0.01},
                {"optimizer": "adam", "learning_rate": 0.01},
            ],
            metric="mse",
            direction="min",
            epochs=3,
        )

        self.assertEqual(len(result["experiments"]), 3)
        self.assertEqual(len(result["ranking"]), 3)
        self.assertIn("best_configuration", result)
        self.assertIn("best_value", result)

        values = [item["value"] for item in result["ranking"]]
        self.assertEqual(values, sorted(values))

        current_weights = [
            list(neuron.weights)
            for layer in model.layers
            for neuron in layer.neurons
        ]

        self.assertEqual(original_weights, current_weights)
        self.assertEqual(alpha.model_count(), 1)


    def test_temporary_model_removed_after_experiment_failure(self):
        alpha = AlphaCore()

        alpha.create_model(
            "modelo_falha",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        engine = UniversalOptimizationEngine(alpha.model_manager)

        with self.assertRaises(OptimizationError):
            engine.optimize(
                "modelo_falha",
                dataset,
                configurations=[
                    {"optimizer": "sgd", "learning_rate": 0.01}
                ],
                metric="perplexity",
                epochs=2,
            )

        self.assertEqual(alpha.model_count(), 1)
        self.assertEqual(
            alpha.list_models(),
            ["modelo_falha"]
        )

    def test_original_model_weights_are_preserved(self):
        alpha = AlphaCore()

        model = alpha.create_model(
            "modelo_original",
            [1, 1],
            activation="linear"
        )

        original_weights = [
            list(neuron.weights)
            for layer in model.layers
            for neuron in layer.neurons
        ]

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        from optimization.universal import OptimizationError

        with self.assertRaises(OptimizationError):
            self.engine.optimize(
                "modelo_original",
                dataset,
                configurations=[
                    {"optimizer": "sgd", "learning_rate": 0.01}
                ]
            )

        current_weights = [
            list(neuron.weights)
            for layer in model.layers
            for neuron in layer.neurons
        ]

        self.assertEqual(original_weights, current_weights)
        self.assertEqual(alpha.model_count(), 1)

    def test_module_import(self):
        self.assertTrue(callable(UniversalOptimizationEngine))

    def test_optimization_error_import(self):
        self.assertTrue(issubclass(OptimizationError, RuntimeError))

    def test_manager_required(self):
        with self.assertRaises(OptimizationError):
            UniversalOptimizationEngine(None)

    def test_empty_configurations_rejected(self):
        with self.assertRaises(OptimizationError):
            self.engine.optimize(
                "modelo",
                [],
                configurations=[]
            )

    def test_invalid_metric_rejected(self):
        with self.assertRaises(OptimizationError):
            self.engine.optimize(
                "modelo",
                [],
                configurations=[
                    {"learning_rate": 0.01}
                ],
                metric="invalid_metric"
            )


    def test_transformer_optimization_with_learning_rates(self):
        from copy import deepcopy

        alpha = AlphaCore()

        texts = [
            "ola mundo alpha",
            "ola alpha mundo",
            "mundo alpha inteligencia",
        ]

        model = alpha.create_language_model(
            "modelo_transformer",
            texts,
            embedding_dim=4,
            hidden_dim=8,
            num_layers=1,
            max_length=8,
        )

        original_weights = deepcopy(model.output_weights)
        original_bias = list(model.output_bias)

        engine = UniversalOptimizationEngine(alpha.model_manager)

        result = engine.optimize(
            "modelo_transformer",
            texts,
            configurations=[
                {"learning_rate": 0.01},
                {"learning_rate": 0.005},
            ],
            metric="loss",
            direction="min",
            epochs=1,
        )

        self.assertEqual(result["architecture"], "transformer_decoder")
        self.assertEqual(len(result["experiments"]), 2)
        self.assertEqual(len(result["ranking"]), 2)

        values = [item["value"] for item in result["ranking"]]
        self.assertEqual(values, sorted(values))

        self.assertEqual(model.output_weights, original_weights)
        self.assertEqual(model.output_bias, original_bias)
        self.assertEqual(alpha.model_count(), 1)
        self.assertEqual(alpha.list_models(), ["modelo_transformer"])

    def test_transformer_rejects_optimizer_configuration(self):
        alpha = AlphaCore()

        texts = [
            "ola mundo alpha",
            "ola alpha mundo",
        ]

        alpha.create_language_model(
            "modelo_transformer_optimizer",
            texts,
            embedding_dim=4,
            hidden_dim=8,
            num_layers=1,
            max_length=8,
        )

        engine = UniversalOptimizationEngine(alpha.model_manager)

        with self.assertRaises(OptimizationError):
            engine.optimize(
                "modelo_transformer_optimizer",
                texts,
                configurations=[
                    {
                        "optimizer": "adam",
                        "learning_rate": 0.01,
                    }
                ],
                metric="loss",
                epochs=1,
            )

        self.assertEqual(alpha.model_count(), 1)
        self.assertEqual(
            alpha.list_models(),
            ["modelo_transformer_optimizer"],
        )


    def test_temporary_model_removed_after_training_failure(self):
        alpha = AlphaCore()

        model = alpha.create_model(
            "modelo_falha_treino",
            [1, 1],
            activation="linear"
        )

        original_weights = [
            list(neuron.weights)
            for layer in model.layers
            for neuron in layer.neurons
        ]

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        engine = UniversalOptimizationEngine(alpha.model_manager)

        def failing_train(*args, **kwargs):
            raise RuntimeError("Falha simulada durante o treino.")

        engine.training_engine.train = failing_train

        with self.assertRaises(OptimizationError):
            engine.optimize(
                "modelo_falha_treino",
                dataset,
                configurations=[
                    {"optimizer": "sgd", "learning_rate": 0.01}
                ],
                metric="mse",
                epochs=2,
            )

        current_weights = [
            list(neuron.weights)
            for layer in model.layers
            for neuron in layer.neurons
        ]

        self.assertEqual(original_weights, current_weights)
        self.assertEqual(alpha.model_count(), 1)
        self.assertEqual(
            alpha.list_models(),
            ["modelo_falha_treino"]
        )


    def test_cleanup_failure_does_not_hide_experiment_failure(self):
        alpha = AlphaCore()

        alpha.create_model(
            "modelo_falha_limpeza",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        manager = alpha.model_manager
        engine = UniversalOptimizationEngine(manager)

        original_train = engine.training_engine.train

        def failing_train(*args, **kwargs):
            raise RuntimeError("Falha original do treino.")

        def failing_remove(name):
            raise RuntimeError("Falha simulada da limpeza.")

        engine.training_engine.train = failing_train
        manager.remove = failing_remove

        with self.assertRaises(OptimizationError) as context:
            engine.optimize(
                "modelo_falha_limpeza",
                dataset,
                configurations=[
                    {"optimizer": "sgd", "learning_rate": 0.01}
                ],
                metric="mse",
                epochs=2,
            )

        self.assertIn("Falha na experiencia", str(context.exception))
        self.assertIsNotNone(original_train)


    def test_partial_duplication_failure_does_not_leave_temporary_model(self):
        alpha = AlphaCore()

        alpha.create_model(
            "modelo_duplicacao_parcial",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        manager = alpha.model_manager
        engine = UniversalOptimizationEngine(manager)
        original_duplicate = manager.duplicate

        def partially_failing_duplicate(name, new_name):
            original_duplicate(name, new_name)
            raise RuntimeError("Falha depois do registo.")

        manager.duplicate = partially_failing_duplicate

        with self.assertRaises(OptimizationError):
            engine.optimize(
                "modelo_duplicacao_parcial",
                dataset,
                configurations=[
                    {"optimizer": "sgd", "learning_rate": 0.01}
                ],
                metric="mse",
                epochs=1,
            )

        self.assertEqual(alpha.model_count(), 1)
        self.assertEqual(
            alpha.list_models(),
            ["modelo_duplicacao_parcial"]
        )


    def test_dense_rejects_transformer_metric_before_duplication(self):
        alpha = AlphaCore()
        alpha.create_model(
            "modelo_metricas_dense",
            [1, 1],
            activation="linear"
        )

        engine = UniversalOptimizationEngine(alpha.model_manager)

        def forbidden_duplicate(*args, **kwargs):
            raise AssertionError("A duplicacao nao devia acontecer.")

        alpha.model_manager.duplicate = forbidden_duplicate

        with self.assertRaises(OptimizationError):
            engine.optimize(
                "modelo_metricas_dense",
                Dataset([([1.0], [2.0])]),
                configurations=[{"learning_rate": 0.01}],
                metric="loss",
                epochs=1,
            )

        self.assertEqual(alpha.model_count(), 1)

    def test_transformer_rejects_dense_metric_before_duplication(self):
        alpha = AlphaCore()
        alpha.create_language_model(
            "modelo_metricas_transformer",
            ["ola mundo alpha", "alpha mundo ola"],
            embedding_dim=4,
            hidden_dim=8,
            num_layers=1,
            max_length=8,
        )

        engine = UniversalOptimizationEngine(alpha.model_manager)

        def forbidden_duplicate(*args, **kwargs):
            raise AssertionError("A duplicacao nao devia acontecer.")

        alpha.model_manager.duplicate = forbidden_duplicate

        with self.assertRaises(OptimizationError):
            engine.optimize(
                "modelo_metricas_transformer",
                ["ola mundo alpha", "alpha mundo ola"],
                configurations=[{"learning_rate": 0.01}],
                metric="mse",
                epochs=1,
            )

        self.assertEqual(alpha.model_count(), 1)


    def test_optimization_report_contains_experiment_numbers_and_summary(self):
        alpha = AlphaCore()

        alpha.create_model(
            "modelo_relatorio",
            [1, 1],
            activation="linear"
        )

        dataset = Dataset([
            ([1.0], [2.0]),
            ([2.0], [4.0]),
        ])

        engine = UniversalOptimizationEngine(alpha.model_manager)

        result = engine.optimize(
            "modelo_relatorio",
            dataset,
            configurations=[
                {"optimizer": "sgd", "learning_rate": 0.01},
                {"optimizer": "adam", "learning_rate": 0.01},
            ],
            metric="mse",
            epochs=2,
        )

        self.assertEqual(
            [item["experiment"] for item in result["experiments"]],
            [1, 2]
        )
        self.assertEqual(result["total_experiments"], 2)
        self.assertEqual(result["completed_experiments"], 2)
        self.assertEqual(
            [item["status"] for item in result["experiments"]],
            ["completed", "completed"]
        )
        self.assertIn("best_configuration", result)
        self.assertIn("best_value", result)


if __name__ == "__main__":
    unittest.main()
