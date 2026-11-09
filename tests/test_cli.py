import os
import tempfile
import unittest

from cli.commands import run
from cli.common import parse_sizes, parse_floats, load_dataset
from cli.interactive import Menu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "examples", "housing.csv")
CORPUS = os.path.join(ROOT, "examples", "corpus_pt.txt")


def capture(argv):
    lines = []
    code = run(argv, output=lines.append)
    return code, "\n".join(lines)


def scripted(answers):
    queue = list(answers)
    out = []

    def fake_input(prompt=""):
        out.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    menu = Menu(input_fn=fake_input, output_fn=out.append)
    menu.run()
    return "\n".join(out)


class TestParsing(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(parse_sizes("2, 4,1"), [2, 4, 1])
        self.assertEqual(parse_floats("1,0.5"), [1.0, 0.5])
        for bad in ("2", "a,b", ""):
            with self.assertRaises(ValueError):
                parse_sizes(bad)
        with self.assertRaises(ValueError):
            parse_floats("x")

    def test_load_dataset(self):
        self.assertEqual(len(load_dataset("xor")), 4)
        self.assertEqual(len(load_dataset(CSV)), 8)
        with self.assertRaises(ValueError):
            load_dataset("nao_existe.csv")


class TestCommands(unittest.TestCase):

    def test_no_command_returns_none(self):
        self.assertIsNone(run([]))

    def test_demo_xor_and_save_predict_info(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "xor.json")
            code, text = capture(["demo", "xor", "--save", path])
            self.assertEqual(code, 0)
            self.assertIn("Perda final", text)

            code, text = capture(["predict", "--model", path, "--input", "1,0"])
            self.assertEqual(code, 0)
            self.assertGreater(float(text.split(":")[1]), 0.8)

            code, text = capture(["predict", "--model", path, "--input", "0,0"])
            self.assertLess(float(text.split(":")[1]), 0.2)

            code, text = capture(["info", "--model", path])
            self.assertIn("rede neuronal", text)

    def test_train_csv_with_validation(self):
        code, text = capture([
            "train", "--data", CSV, "--layers", "2,4,1", "--epochs", "30",
            "--validation", "0.25", "--lr", "0.02"
        ])
        self.assertEqual(code, 0)
        self.assertIn("validacao", text)

    def test_train_checks_layer_sizes(self):
        code, _ = capture(["train", "--data", "xor", "--layers", "3,2,1"])
        self.assertEqual(code, 1)
        code, _ = capture(["train", "--data", "xor", "--layers", "2,2,2"])
        self.assertEqual(code, 1)

    def test_bad_inputs_return_error_code(self):
        self.assertEqual(capture(["predict", "--model", "nada.json",
                                  "--input", "1"])[0], 1)
        self.assertEqual(capture(["train", "--data", "xor", "--layers", "2",
                                  ])[0], 1)

    def test_lm_train_and_generate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "lm.json")
            code, text = capture([
                "lm-train", "--text", CORPUS, "--epochs", "30",
                "--prompt", "o gato", "--save", path
            ])
            self.assertEqual(code, 0)
            self.assertIn("perplexidade", text)

            code, text = capture([
                "generate", "--model", path, "--prompt", "o gato",
                "--temperature", "0", "--tokens", "6"
            ])
            self.assertEqual(code, 0)
            self.assertTrue(text.startswith("o gato"))

            # Um modelo de linguagem nao serve para 'predict'.
            self.assertEqual(capture(["predict", "--model", path,
                                      "--input", "1"])[0], 1)


class TestMenu(unittest.TestCase):

    def test_full_session(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.json")
            text = scripted([
                "1", "rede", "2,4,1", "tanh", "sigmoid",   # criar
                "4", "xor", "xor",                          # dataset
                "5", "", "", "none", "adam", "0.05", "400", "1",  # treinar
                "6", "", "",                                # avaliar
                "7", "", "1,0",                             # prever
                "8", "", path,                              # guardar
                "3", "rede",                                # remover
                "9", path, "carregada",                     # carregar
                "2", "12", "0",
            ])
        self.assertIn("Modelo criado", text)
        self.assertIn("Perda final", text)
        self.assertIn("MSE", text)
        self.assertIn("Modelo guardado", text)
        self.assertIn("Modelo 'carregada' carregado", text)
        self.assertIn("carregada", text)
        self.assertNotIn("Erro", text)

    def test_language_model_session(self):
        text = scripted([
            "10", "lm", CORPUS, "5", "0.02",
            "11", "", "o gato", "5", "0",
            "0",
        ])
        self.assertIn("perplexidade", text)
        self.assertNotIn("Erro", text)

    def test_errors_do_not_crash(self):
        text = scripted([
            "5",              # sem modelos
            "99",             # opcao invalida
            "1", "m", "abc",  # tamanhos invalidos
            "3", "nada",      # remover inexistente
            "0",
        ])
        self.assertGreaterEqual(text.count("Erro"), 3)
        self.assertIn("Opcao invalida", text)

    def test_incompatible_model_and_dataset(self):
        text = scripted([
            "1", "m", "3,2,1", "tanh", "linear",
            "4", "xor", "xor",
            "5", "", "", "adam", "", "5", "1",
            "0",
        ])
        self.assertIn("incompativel", text)

    def test_eof_exits_cleanly(self):
        text = scripted([])
        self.assertIn("encerrado", text)


if __name__ == "__main__":
    unittest.main()


class TestCliClassificationAndScaling(unittest.TestCase):

    FLORES = os.path.join(ROOT, "examples", "flores.csv")

    def test_demo_blobs(self):
        code, text = capture(["demo", "blobs"])
        self.assertEqual(code, 0)
        self.assertIn("Acuracia", text)
        self.assertIn("Matriz de confusao", text)

    def test_train_classification_save_predict_info(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "flores.json")
            code, text = capture([
                "train", "--data", self.FLORES, "--task", "classification",
                "--layers", "2,6,3", "--epochs", "200", "--lr", "0.03",
                "--save", path
            ])
            self.assertEqual(code, 0)
            self.assertIn("Acuracia", text)

            code, text = capture(["predict", "--model", path,
                                  "--input", "7.4,3.0"])
            self.assertEqual(code, 0)
            self.assertIn("Classe prevista: grande", text)

            code, text = capture(["predict", "--model", path,
                                  "--input", "4.8,3.3"])
            self.assertIn("Classe prevista: pequena", text)

            code, text = capture(["info", "--model", path])
            self.assertIn("classes: grande, media, pequena", text)
            self.assertIn("normalizacao: standard", text)

    def test_classification_layer_hint(self):
        code, _ = capture(["train", "--data", self.FLORES, "--task",
                           "classification", "--layers", "2,4,2"])
        self.assertEqual(code, 1)

    def test_regression_with_scaling_predicts_original_units(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "casas.json")
            code, _ = capture([
                "train", "--data", CSV, "--layers", "2,6,1", "--epochs", "300",
                "--lr", "0.02", "--scale", "minmax", "--scale-targets",
                "--save", path
            ])
            self.assertEqual(code, 0)
            code, text = capture(["predict", "--model", path,
                                  "--input", "0.9,0.6"])
            self.assertAlmostEqual(float(text.split(":")[1]), 0.70, delta=0.1)

    def test_task_mismatch_with_sample(self):
        code, _ = capture(["train", "--data", "xor", "--task",
                           "classification", "--layers", "2,2,1"])
        self.assertEqual(code, 1)


class TestMenuClassification(unittest.TestCase):

    def test_classification_session(self):
        text = scripted([
            "1", "c", "2,8,3", "tanh", "linear",
            "4", "blobs", "blobs",
            "5", "", "", "none", "adam", "0.02", "40", "8",
            "6", "", "",
            "7", "", "-1,-1",
            "0",
        ])
        self.assertIn("Acuracia no treino", text)
        self.assertIn("Matriz de confusao", text)
        self.assertIn("Classe prevista: A", text)
        self.assertNotIn("Erro", text)

    def test_classification_requires_linear_output(self):
        text = scripted([
            "1", "c", "2,8,3", "tanh", "sigmoid",
            "4", "blobs", "blobs",
            "5", "", "",
            "0",
        ])
        self.assertIn("Erro: Para classificar", text)
