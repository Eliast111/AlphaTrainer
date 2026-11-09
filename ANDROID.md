# AlphaTrainer no Android

A interface Android usa Kivy e reutiliza o motor Python puro do projeto.
O APK **não está pré-compilado**: a compilação Android requer Linux (ou WSL),
Java, Android SDK/NDK e Buildozer. A preparação do código pode ser feita assim:

```bash
python android_app/prepare_android.py
cd android_app
pip install buildozer
buildozer android debug
```

O APK de depuração será criado em `android_app/bin/` quando o Buildozer terminar.
Em Android/Termux, a compilação pode ser limitada; recomenda-se Ubuntu/WSL ou
Linux. Para experimentar sem compilar APK, use Pydroid 3 com Kivy instalado,
ou utilize o dashboard web local descrito em `WEB.md`.

A primeira compilação descarrega componentes do Android e pode demorar bastante.


O APK não inclui SciPy/Matplotlib por padrão para evitar builds muito pesados.
O motor usa fallback Python puro; NumPy pode ser adicionado ao Buildozer apenas
quando a receita/arquitetura Android usada for compatível. Consulte TERMUX.md
para usar as bibliotecas instaladas no Termux (fora do APK).
