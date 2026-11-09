"""
AlphaTrainer 1.1
Ponto de entrada principal.

  python main.py                  menu interativo
  python main.py demo xor         exemplo pronto a usar
  python main.py train --help     treinar com os teus dados
  python main.py lm-train --help  modelo de linguagem (Transformer)
"""

import sys

from cli.commands import run
from cli.interactive import Menu


def main(argv=None):
    code = run(argv)

    if code is not None:
        return code

    try:
        Menu().run()
    except KeyboardInterrupt:
        print("\nAlphaTrainer encerrado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
