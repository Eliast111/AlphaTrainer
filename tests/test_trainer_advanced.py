import unittest

from data.samples import load_sample
from learning.optimizer import Adam, GradientDescent
from models.factory import create_model
from training.trainer import Trainer


class TestTrainerAdvanced(unittest.TestCase):

    def test_xor_is_learned(self):
        model = create_model(
            [2, 4, 1], activation="tanh", output_activation="sigmoid"
        )
        data = load_sample("xor")
        Trainer(Adam(0.05)).fit(model, data, epochs=500)
        for inputs, targets in data:
            self.assertAlmostEqual(
                round(model.forward(inputs)[0]), targets[0]
            )

    def test_mini_batch(self):
        model = create_model([1, 1], activation="linear")
        data = load_sample("linear")
        result = Trainer(Adam(0.05)).fit(model, data, epochs=300, batch_size=5)
        self.assertLess(result["final_loss"], 1e-3)

    def test_validation_history(self):
        data = load_sample("seno")
        train, validation = data.split(0.25, seed=1)
        model = create_model([1, 4, 1], activation="tanh", output_activation="linear")
        result = Trainer(Adam(0.02)).fit(
            model, train, epochs=20, validation_data=validation
        )
        self.assertEqual(len(result["val_history"]), 20)
        self.assertIn("final_val_loss", result)

    def test_early_stopping_restores_best(self):
        model = create_model([1, 1], activation="linear")
        data = load_sample("linear")
        result = Trainer(GradientDescent(0.05)).fit(
            model, data, epochs=5000, patience=3, min_delta=1e-3
        )
        self.assertTrue(result["stopped_early"])
        self.assertLess(result["epochs_run"], 5000)
        self.assertIn("best_epoch", result)

    def test_target_loss(self):
        model = create_model([1, 1], activation="linear")
        result = Trainer(Adam(0.1)).fit(
            model, load_sample("linear"), epochs=2000, target_loss=1e-4
        )
        self.assertTrue(result["stopped_early"])
        self.assertLessEqual(result["final_loss"], 1e-4)

    def test_callback_can_stop(self):
        model = create_model([1, 1], activation="linear")
        seen = []

        def callback(epoch, loss, val):
            seen.append(epoch)
            return epoch >= 3

        result = Trainer(Adam(0.1)).fit(
            model, load_sample("linear"), epochs=50, callback=callback
        )
        self.assertEqual(seen, [1, 2, 3])
        self.assertEqual(result["epochs_run"], 3)

    def test_clip_norm_runs(self):
        model = create_model([1, 1], activation="linear")
        result = Trainer(GradientDescent(0.05)).fit(
            model, load_sample("linear"), epochs=10, clip_norm=0.5
        )
        self.assertEqual(result["epochs_run"], 10)

    def test_reproducible_with_seed(self):
        results = []
        for _ in range(2):
            model = create_model([1, 3, 1], activation="tanh", seed=5)
            results.append(Trainer(Adam(0.05)).fit(
                model, load_sample("seno"), epochs=5, seed=3
            )["history"])
        self.assertEqual(results[0], results[1])

    def test_invalid_options(self):
        trainer = Trainer(Adam(0.1))
        model = create_model([1, 1])
        data = load_sample("linear")
        for bad in ({"batch_size": 0}, {"patience": 0},
                    {"clip_norm": -1}, {"callback": 5}):
            with self.assertRaises((ValueError, TypeError)):
                trainer.fit(model, data, epochs=2, **bad)

    def test_incompatible_validation_data(self):
        trainer = Trainer(Adam(0.1))
        model = create_model([1, 1])
        with self.assertRaises(ValueError):
            trainer.fit(model, load_sample("linear"), epochs=2,
                        validation_data=load_sample("xor"))


if __name__ == "__main__":
    unittest.main()
