import unittest

from neural.neuron import Neuron
from neural.layer import Layer
from neural.network import Network
from learning.optimizer import GradientDescent
from training.trainer import Trainer


class TestAlphaTrainer(unittest.TestCase):

    def create_network(self):
        return Network([
            Layer([
                Neuron([0.0], bias=0.0, activation="linear")
            ])
        ])

    def test_training_reduces_error(self):
        network = self.create_network()

        trainer = Trainer(
            GradientDescent(learning_rate=0.1)
        )

        result = trainer.fit(
            network,
            [([1.0], [2.0])],
            epochs=100
        )

        self.assertLess(result["final_loss"], 0.001)

    def test_training_history(self):
        trainer = Trainer(GradientDescent(0.1))

        result = trainer.fit(
            self.create_network(),
            [([1.0], [2.0])],
            epochs=5
        )

        self.assertEqual(result["epochs"], 5)
        self.assertEqual(len(result["history"]), 5)

    def test_empty_dataset(self):
        trainer = Trainer(GradientDescent(0.1))

        with self.assertRaises(ValueError):
            trainer.fit(self.create_network(), [], epochs=5)

    def test_invalid_epochs(self):
        trainer = Trainer(GradientDescent(0.1))

        with self.assertRaises(ValueError):
            trainer.fit(
                self.create_network(),
                [([1.0], [2.0])],
                epochs=0
            )

    def test_invalid_optimizer(self):
        with self.assertRaises(TypeError):
            Trainer(None)


if __name__ == "__main__":
    unittest.main()
