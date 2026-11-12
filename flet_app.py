import asyncio
import flet as ft

from core import AlphaCore
from data.samples import available_samples, load_sample
from learning.optimizer import create_optimizer


def main(page: ft.Page):
    page.title = "AlphaTrainer AI Studio"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0B1020"
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

    alpha = AlphaCore()

    sample = ft.Dropdown(
        label="Dataset de treino",
        value="xor",
        width=350,
        options=[
            ft.DropdownOption(key=name, text=description)
            for name, description in available_samples().items()
        ],
    )

    model_name = ft.TextField(
        label="Nome do modelo",
        value="demo_xor",
        width=350,
    )

    inputs = ft.TextField(
        label="Entradas para previsão",
        value="0,1",
        width=350,
        hint_text="Exemplo: 0,1",
    )

    status = ft.Text(
        "AlphaTrainer pronto para trabalhar.",
        color="#A5B4FC",
        selectable=True,
    )

    progress = ft.ProgressRing(
        visible=False,
        width=22,
        height=22,
    )

    models_view = ft.Column(spacing=5)

    def refresh_models():
        names = alpha.list_models()
        models_view.controls.clear()

        if not names:
            models_view.controls.append(
                ft.Text("Ainda não existem modelos treinados.")
            )
        else:
            for name in names:
                models_view.controls.append(
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.PSYCHOLOGY),
                        title=ft.Text(name),
                    )
                )

        page.update()

    async def train_clicked(e):
        name = model_name.value.strip()
        key = sample.value

        if not name or not key:
            status.value = "Preenche o nome e seleciona um dataset."
            page.update()
            return

        progress.visible = True
        status.value = "A preparar o treino..."
        train_button.disabled = True
        page.update()

        def train_model():
            dataset = load_sample(key)

            if name in alpha.list_models():
                alpha.remove_model(name)

            output = (
                "sigmoid"
                if key in ("xor", "and", "or")
                else "linear"
            )

            alpha.create_model(
                name,
                [dataset.input_size, 4, dataset.target_size],
                activation="tanh",
                output_activation=output,
                seed=42,
            )

            result = alpha.train_model(
                name,
                dataset,
                create_optimizer("adam", 0.05),
                epochs=250,
                batch_size=1,
                shuffle=True,
                seed=42,
            )

            return result

        try:
            result = await asyncio.to_thread(train_model)

            loss = result.get("final_loss", 0)

            status.value = (
                f"Treino concluído!\n"
                f"Modelo: {name}\n"
                f"Perda final: {loss:.6f}"
            )

            refresh_models()

        except Exception as exc:
            status.value = f"Erro no treino: {exc}"

        finally:
            progress.visible = False
            train_button.disabled = False
            page.update()

    async def predict_clicked(e):
        name = model_name.value.strip()

        try:
            values = [
                float(v.strip())
                for v in inputs.value.split(",")
            ]

            if not name:
                raise ValueError("Indica o nome do modelo.")

            result = await asyncio.to_thread(
                alpha.predict,
                name,
                values,
            )

            status.value = (
                "Previsão: "
                + ", ".join(f"{v:.6f}" for v in result)
            )

        except Exception as exc:
            status.value = f"Erro na previsão: {exc}"

        page.update()

    def list_clicked(e):
        refresh_models()

    train_button = ft.FilledButton(
        "Treinar modelo",
        icon=ft.Icons.PLAY_ARROW,
        on_click=train_clicked,
    )

    predict_button = ft.OutlinedButton(
        "Fazer previsão",
        icon=ft.Icons.AUTO_GRAPH,
        on_click=predict_clicked,
    )

    page.add(
        ft.Column(
            controls=[
                ft.Text(
                    "AlphaTrainer",
                    size=30,
                    weight=ft.FontWeight.BOLD,
                    color="#A78BFA",
                ),
                ft.Text(
                    "AI Studio | Inteligência Artificial em Python",
                    color="#94A3B8",
                ),
                ft.Divider(),
                ft.Text(
                    "TREINO DE REDES NEURONAIS",
                    size=16,
                    weight=ft.FontWeight.BOLD,
                ),
                sample,
                model_name,
                ft.Row(
                    controls=[train_button, progress],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(),
                ft.Text(
                    "PREVISÕES",
                    size=16,
                    weight=ft.FontWeight.BOLD,
                ),
                inputs,
                predict_button,
                ft.Divider(),
                ft.Text(
                    "ESTADO DO SISTEMA",
                    weight=ft.FontWeight.BOLD,
                ),
                status,
                ft.Divider(),
                ft.Text(
                    "MODELOS DISPONÍVEIS",
                    weight=ft.FontWeight.BOLD,
                ),
                ft.OutlinedButton(
                    "Atualizar modelos",
                    icon=ft.Icons.REFRESH,
                    on_click=list_clicked,
                ),
                models_view,
            ],
            spacing=14,
            expand=True,
        )
    )

    refresh_models()


if __name__ == "__main__":
    ft.run(main)
