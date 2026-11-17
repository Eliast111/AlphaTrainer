# SDK local Python — AlphaTrainer

O SDK é uma fachada Python síncrona sobre `AlphaCore`. Não inicia servidor,
não usa HTTP e não envia dados para serviços externos.

```python
from core.alpha_core import AlphaCore
from sdk import LocalLearningSDK
from learning.optimizer import Adam

core = AlphaCore()
sdk = LocalLearningSDK(core)

sdk.create_dense_model("Exemplo", [2, 3, 1])
resultado = sdk.train(
    "Exemplo",
    [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0])],
    optimizer=Adam(learning_rate=0.01),
    epochs=10,
    checkpoint_path="persistence/sessions/exemplo.json",
)
previsao = sdk.predict("Exemplo", [0.0, 1.0])
```

Para continuar a sessão, chamar `sdk.train` com o mesmo dataset e opções
compatíveis, aumentar `epochs` para o total pretendido e passar `resume=True`.
A retoma é suportada apenas por adaptadores que declarem `resume=True`.
```
