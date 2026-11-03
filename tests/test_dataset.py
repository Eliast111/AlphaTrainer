import unittest

from data.dataset import Dataset


class TestAlphaDataset(unittest.TestCase):

    def setUp(self):
        self.samples = [
            ([1.0, 2.0], [3.0]),
            ([2.0, 3.0], [5.0]),
            ([3.0, 4.0], [7.0]),
            ([4.0, 5.0], [9.0]),
            ([5.0, 6.0], [11.0])
        ]

    def test_dataset_creation(self):
        dataset = Dataset(self.samples)
        self.assertEqual(len(dataset), 5)

    def test_dataset_dimensions(self):
        dataset = Dataset(self.samples)

        self.assertEqual(dataset.input_size, 2)
        self.assertEqual(dataset.target_size, 1)

    def test_dataset_iteration(self):
        dataset = Dataset(self.samples)

        self.assertEqual(len(list(dataset)), 5)

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            Dataset([])

    def test_inconsistent_input_size(self):
        invalid = [
            ([1.0, 2.0], [3.0]),
            ([2.0], [4.0])
        ]

        with self.assertRaises(ValueError):
            Dataset(invalid)

    def test_non_numeric_data(self):
        invalid = [
            (["abc"], [2.0])
        ]

        with self.assertRaises(ValueError):
            Dataset(invalid)

    def test_invalid_numeric_values(self):
        invalid = [
            ([float("nan")], [2.0])
        ]

        with self.assertRaises(ValueError):
            Dataset(invalid)

    def test_train_validation_split(self):
        dataset = Dataset(self.samples)

        training, validation = dataset.split(
            validation_ratio=0.2,
            shuffle=False
        )

        self.assertEqual(len(training), 4)
        self.assertEqual(len(validation), 1)

    def test_split_preserves_all_samples(self):
        dataset = Dataset(self.samples)

        training, validation = dataset.split(seed=42)

        self.assertEqual(
            len(training) + len(validation),
            len(dataset)
        )

    def test_reproducible_shuffle(self):
        dataset = Dataset(self.samples)

        first = dataset.split(seed=42)
        second = dataset.split(seed=42)

        self.assertEqual(first[0].samples, second[0].samples)
        self.assertEqual(first[1].samples, second[1].samples)

    def test_invalid_split_ratio(self):
        dataset = Dataset(self.samples)

        with self.assertRaises(ValueError):
            dataset.split(validation_ratio=1.0)

    def test_split_requires_two_samples(self):
        dataset = Dataset([([1.0], [2.0])])

        with self.assertRaises(ValueError):
            dataset.split()


if __name__ == "__main__":
    unittest.main()
