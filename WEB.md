# AlphaTrainer Studio (dashboard web local)

O dashboard é opcional e não altera as dependências do motor principal.

```bash
python -m pip install -r requirements-web.txt
python webapp.py
```

Abra `http://127.0.0.1:8765` no navegador do mesmo dispositivo.
O servidor fica limitado a localhost por segurança. Não use o servidor Flask de
desenvolvimento exposto à internet. Modelos guardados ficam em `saved_models/`.
