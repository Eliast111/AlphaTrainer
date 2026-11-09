"""Graficos opcionais de treino usando Matplotlib."""
from pathlib import Path

def save_loss_plot(history, path, validation_history=None, title="Curva de aprendizagem"):
    if not history:
        raise ValueError("O historico de treino esta vazio.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Matplotlib nao esta disponivel neste Python.") from exc
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(history) + 1), history, label="Treino", linewidth=2)
    if validation_history:
        ax.plot(range(1, len(validation_history) + 1), validation_history, label="Validacao", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Perda")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return str(destination)
