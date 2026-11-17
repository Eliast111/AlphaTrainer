import os
import json
import tempfile
import unittest

from learning.optimizer import GradientDescent, Momentum, Adam
from training.trainer import Trainer
from training.session_checkpoint import load_training_session
from tests.test_optimizer_state import make_network


class TestTrainingSessionCheckpoint(unittest.TestCase):

    def setUp(self):
        self.dataset = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]

    def optimizer_cases(self):
        return [
            ("SGD", lambda: GradientDescent(learning_rate=0.01)),
            ("Momentum", lambda: Momentum(
                learning_rate=0.01,
                momentum=0.9,
            )),
            ("Adam", lambda: Adam(
                learning_rate=0.01,
            )),
        ]

    def assert_networks_equal(self, first, second):
        self.assertEqual(len(first.layers), len(second.layers))

        for first_layer, second_layer in zip(
            first.layers, second.layers
        ):
            self.assertEqual(
                len(first_layer.neurons),
                len(second_layer.neurons),
            )

            for first_neuron, second_neuron in zip(
                first_layer.neurons,
                second_layer.neurons,
            ):
                self.assertEqual(
                    first_neuron.weights,
                    second_neuron.weights,
                )
                self.assertEqual(
                    first_neuron.bias,
                    second_neuron.bias,
                )

    def test_resume_matches_continuous_training_for_all_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:

            for name, create_optimizer in self.optimizer_cases():
                with self.subTest(optimizer=name):
                    continuous_network = make_network()
                    resumed_network = make_network()

                    continuous_trainer = Trainer(create_optimizer())
                    resumed_trainer = Trainer(create_optimizer())

                    continuous = continuous_trainer.fit(
                        continuous_network,
                        self.dataset,
                        epochs=4,
                        batch_size=2,
                        shuffle=True,
                        seed=17,
                    )

                    path = os.path.join(
                        directory,
                        name.lower() + "_session.json",
                    )

                    resumed_trainer.fit(
                        resumed_network,
                        self.dataset,
                        epochs=2,
                        batch_size=2,
                        shuffle=True,
                        seed=17,
                        checkpoint_path=path,
                    )

                    resumed = resumed_trainer.fit(
                        resumed_network,
                        self.dataset,
                        epochs=4,
                        batch_size=2,
                        shuffle=True,
                        seed=17,
                        checkpoint_path=path,
                        resume=True,
                    )

                    self.assertEqual(
                        continuous["history"],
                        resumed["history"],
                    )

                    self.assert_networks_equal(
                        continuous_network,
                        resumed_network,
                    )

    def test_rejects_different_dataset(self):
        network = make_network()
        trainer = Trainer(Adam(learning_rate=0.01))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "session.json")

            trainer.fit(
                network,
                self.dataset,
                epochs=2,
                checkpoint_path=path,
            )

            different_dataset = [
                ([0.0, 0.0], [1.0]),
                ([0.0, 1.0], [0.0]),
            ]

            with self.assertRaises(ValueError):
                trainer.fit(
                    network,
                    different_dataset,
                    epochs=4,
                    checkpoint_path=path,
                    resume=True,
                )


    def test_rejects_unknown_checkpoint_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.json")

            with open(path, "w", encoding="utf-8") as file:
                file.write('{"format": "unknown_format"}')

            with self.assertRaises(ValueError):
                load_training_session(path)

    def test_rejects_corrupted_checkpoint_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corrupted.json")

            with open(path, "w", encoding="utf-8") as file:
                file.write('{"format": ')

            with self.assertRaises(ValueError):
                load_training_session(path)


    def _assert_checkpoint_rejected(self, field, invalid_value):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "session.json")

            Trainer(Adam(learning_rate=0.01)).fit(
                make_network(),
                self.dataset,
                epochs=2,
                checkpoint_path=path,
            )

            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            payload[field] = invalid_value

            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file)

            with self.assertRaises(ValueError):
                Trainer(Adam(learning_rate=0.01)).fit(
                    make_network(),
                    self.dataset,
                    epochs=4,
                    checkpoint_path=path,
                    resume=True,
                )

    def test_rejects_invalid_training_history(self):
        self._assert_checkpoint_rejected("history", "invalid")

    def test_rejects_invalid_sample_order(self):
        self._assert_checkpoint_rejected("order", [0, 0, 2, 3])

    def test_rejects_invalid_network_state(self):
        self._assert_checkpoint_rejected("network_state", [])

    def test_rejects_invalid_best_epoch(self):
        self._assert_checkpoint_rejected("best_epoch", -1)

    def test_rejects_invalid_best_loss(self):
        self._assert_checkpoint_rejected("best_loss", float("nan"))

    def test_rejects_invalid_stale_counter(self):
        self._assert_checkpoint_rejected("stale", -1)

    def test_rejects_invalid_rng_state(self):
        self._assert_checkpoint_rejected("rng_state", ["invalid"])

    def test_invalid_rng_does_not_mutate_network_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "session.json")
            source_network = make_network()
            Trainer(Adam(learning_rate=0.01)).fit(
                source_network, self.dataset, epochs=2, checkpoint_path=path
            )

            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            payload["rng_state"] = ["invalid"]
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file)

            network = make_network()
            optimizer = Adam(learning_rate=0.02)
            trainer = Trainer(optimizer)
            before_network = make_network()
            before_optimizer = optimizer.state_dict(network)

            with self.assertRaises(ValueError):
                trainer.fit(
                    network, self.dataset, epochs=4,
                    checkpoint_path=path, resume=True
                )

            self.assert_networks_equal(network, before_network)
            self.assertEqual(optimizer.state_dict(network), before_optimizer)



if __name__ == "__main__":
    unittest.main()
