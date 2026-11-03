import unittest
import tempfile
import os

from data.dataset import Dataset


class TestAlphaDatasetCSV(unittest.TestCase):

    def create_csv(self, content):
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            encoding="utf-8",
            delete=False
        )

        temporary.write(content)
        temporary.close()

        self.addCleanup(
            lambda: os.path.exists(temporary.name)
            and os.remove(temporary.name)
        )

        return temporary.name

    def test_csv_import(self):
        path = self.create_csv(
            "entrada1,entrada2,resultado\n"
            "1,2,3\n"
            "2,3,5\n"
        )

        dataset = Dataset.from_csv(path)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], ([1.0, 2.0], [3.0]))

    def test_csv_without_header(self):
        path = self.create_csv("1,2\n3,4\n")

        dataset = Dataset.from_csv(path, has_header=False)

        self.assertEqual(len(dataset), 2)

    def test_custom_delimiter(self):
        path = self.create_csv(
            "entrada;resultado\n"
            "5;10\n"
            "6;12\n"
        )

        dataset = Dataset.from_csv(path, delimiter=";")

        self.assertEqual(dataset[0], ([5.0], [10.0]))

    def test_multiple_target_columns(self):
        path = self.create_csv(
            "a,b,c,d\n"
            "1,2,3,4\n"
        )

        dataset = Dataset.from_csv(path, target_columns=2)

        self.assertEqual(dataset[0], ([1.0, 2.0], [3.0, 4.0]))

    def test_invalid_target_columns(self):
        path = self.create_csv("a,b\n1,2\n")

        with self.assertRaises(ValueError):
            Dataset.from_csv(path, target_columns=0)

    def test_non_numeric_csv(self):
        path = self.create_csv(
            "entrada,resultado\n"
            "abc,10\n"
        )

        with self.assertRaises(ValueError):
            Dataset.from_csv(path)

    def test_empty_csv(self):
        path = self.create_csv("entrada,resultado\n")

        with self.assertRaises(ValueError):
            Dataset.from_csv(path)

    def test_inconsistent_columns(self):
        path = self.create_csv(
            "a,b,c\n"
            "1,2,3\n"
            "4,5\n"
        )

        with self.assertRaises(ValueError):
            Dataset.from_csv(path)


if __name__ == "__main__":
    unittest.main()
