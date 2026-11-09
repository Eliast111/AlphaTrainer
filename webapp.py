"""Optional local web dashboard for AlphaTrainer. Install Flask via requirements-web.txt."""
from pathlib import Path
import secrets
from flask import Flask, flash, redirect, render_template, request, url_for

from core import AlphaCore
from data.samples import available_samples, load_sample
from learning.optimizer import create_optimizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
app = Flask(__name__, template_folder="web/templates")
app.secret_key = secrets.token_hex(32)
alpha = AlphaCore()


def _model_path(name):
    # Accept a simple filename only; never allow path traversal.
    if not name or Path(name).name != name or not name.endswith(".json"):
        raise ValueError("Nome de ficheiro inválido.")
    return MODEL_DIR / name


@app.get("/")
def index():
    models = []
    for name in alpha.list_models():
        models.append(alpha.describe_model(name))
    return render_template("dashboard.html", models=models,
                           samples=available_samples())


@app.post("/demo")
def demo():
    sample = request.form.get("sample", "xor")
    if sample not in available_samples():
        flash("Dataset de exemplo desconhecido.", "error")
        return redirect(url_for("index"))
    if "demo_" + sample in alpha.list_models():
        alpha.remove_model("demo_" + sample)
    dataset = load_sample(sample)
    sizes = [dataset.input_size, 4, dataset.target_size]
    output_activation = "sigmoid" if sample in ("xor", "and", "or") else "linear"
    alpha.create_model("demo_" + sample, sizes, activation="tanh",
                       output_activation=output_activation, seed=42)
    result = alpha.train_model("demo_" + sample, dataset,
                               create_optimizer("adam", 0.05), epochs=250,
                               batch_size=1, shuffle=True, seed=42)
    flash(f"Treino concluído: {result.get('final_loss', 0):.6f} de perda final.", "success")
    return redirect(url_for("index"))


@app.post("/predict")
def predict():
    name = request.form.get("model", "")
    raw = request.form.get("inputs", "")
    try:
        model = alpha.get_model(name)
        if not hasattr(model, "input_size"):
            raise ValueError("Este modelo não aceita entradas numéricas diretas.")
        values = [float(v.strip()) for v in raw.split(",")]
        if not values or len(values) != model.input_size():
            raise ValueError("Número de entradas incorreto.")
        result = alpha.predict(name, values)
        flash("Previsão: " + ", ".join(f"{v:.5f}" for v in result), "success")
    except (ValueError, KeyError, TypeError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("index"))


@app.post("/save")
def save():
    name = request.form.get("model", "")
    filename = request.form.get("filename", "")
    try:
        _model_path(filename)
        alpha.save_model(name, _model_path(filename))
        flash(f"Modelo guardado em saved_models/{filename}.", "success")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("index"))


@app.post("/load")
def load():
    name = request.form.get("name", "")
    filename = request.form.get("filename", "")
    try:
        alpha.load_model(name, _model_path(filename))
        flash(f"Modelo {name} carregado.", "success")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Local-only by default. Do not expose the development server to the internet.
    app.run(host="127.0.0.1", port=8765, debug=False)
